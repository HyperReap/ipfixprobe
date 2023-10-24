/**
 * \file neuron_plugin.cpp
 * \brief Plugin for parsing neuron_plugin traffic.
 * \author PETR URBANEK urbanek.vk@gmail.com
 * \date 2023
 */
/*
 * Copyright (C) 2023 CESNET
 *
 * LICENSE TERMS
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name of the Company nor the names of its contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 *
 *
 */

#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

#include "neuron_plugin.hpp"
using namespace torch::autograd;

namespace ipxp {

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int neuronRecord::REGISTERED_ID = -1;

__attribute__((constructor)) static void register_this_plugin()
{
    static PluginRecord rec = PluginRecord("neuron", []() { return new NEURON_PLUGINPlugin(); });
    register_plugin(&rec);
    neuronRecord::REGISTERED_ID = register_extension();
}

NEURON_PLUGINPlugin::NEURON_PLUGINPlugin() {}

NEURON_PLUGINPlugin::~NEURON_PLUGINPlugin()
{
    close();
}

void NEURON_PLUGINPlugin::init(const char* params) {}

void NEURON_PLUGINPlugin::close() {}

ProcessPlugin* NEURON_PLUGINPlugin::copy()
{
    return new NEURON_PLUGINPlugin(*this);
}

int NEURON_PLUGINPlugin::pre_create(Packet& pkt)
{
    return 0;
}

int NEURON_PLUGINPlugin::post_create(Flow& rec, const Packet& pkt)
{
    neuronRecord* record = new neuronRecord(); // created new flow, can keep 30 packets from it
    rec.add_extension(record); // register this flow
    printf("Flow Create: #%d \n", record->REGISTERED_ID);
    update_record(record, pkt);

    return 0;
}

int NEURON_PLUGINPlugin::pre_update(Flow& rec, Packet& pkt)
{
    return 0;
}

// catch 30 packets of first 100 bytes, jak sakra funguje tohle vzbirani flow :D proc jsou vsechnz flow #23, ID pluginu? TODO::
int NEURON_PLUGINPlugin::post_update(Flow& rec, const Packet& pkt)
{
    neuronRecord* data = static_cast<neuronRecord*>(rec.get_extension(neuronRecord::REGISTERED_ID));
    printf("Flow Update: #%d ", data->REGISTERED_ID);
    data->order++;
    printf("Order is now: %d\n", data->order);

    printf("\n");
    if (data->order > BUFFER_COUNT) {
        printf("Too many packets in this flow > %d\n", BUFFER_COUNT);
        return 0;
    }

    update_record(data, pkt);

    return 0;
}

void NEURON_PLUGINPlugin::update_record(neuronRecord* data, const Packet& pkt)
{
    // zatim seru an smer
    printf("incoming packet.len %d \n", pkt.packet_len);
    printf("incoming payload.len %d \n", pkt.payload_len);
    //  data->packets[data->order].size = MIN(CONTENT_SIZE, pkt.payload_len); // watchout for
    //  overflow
    data->packets[data->order].size
        = pkt.payload_len < CONTENT_SIZE ? pkt.payload_len : CONTENT_SIZE;
    printf("size saved on order %d = %d \n", data->order, data->packets[data->order].size);
    printf("\n");

    memcpy(
        data->packets[data->order].data,
        pkt.payload,
        data->packets[data->order].size); // just copy first CONTENT_SIZE packets
}

void NEURON_PLUGINPlugin::pre_export(Flow& rec)
{
    printf("EXPORT:\n");

    auto size_tensor = torch::tensor({});
    neuronRecord* data = static_cast<neuronRecord*>(rec.get_extension(neuronRecord::REGISTERED_ID));

    for (size_t i = 0; i < BUFFER_COUNT; i++) {
        auto size = data->packets[i].size;
        size_tensor = torch::cat({size_tensor, torch::tensor({size})}, 0);
    }

    std::cout << std::endl << "SIZE TENSOR" << "\t" << size_tensor << std::endl;

    runNN(size_tensor);
}

void NEURON_PLUGINPlugin::runNN(torch::Tensor input)
{
    torch::jit::script::Module module = this->LoadModel();
    // Create an example input tensor
    //  torch::Tensor input = torch::randn({1, 30});

    // Run inference on the CPU
    torch::Tensor output = module.forward({input}).toTensor();

    // Print the output
    std::cout << "Output: " << output << std::endl;
}

// Load torchscript model from file
torch::jit::script::Module NEURON_PLUGINPlugin::LoadModel()
{
    torch::jit::script::Module loaded_model;
    try {
        loaded_model = torch::jit::load("../tmp/scripted_model.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw e;
    }

    return loaded_model;
}

} // namespace ipxp
