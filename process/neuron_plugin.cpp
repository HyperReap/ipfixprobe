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

NEURON_PLUGINPlugin::NEURON_PLUGINPlugin() 
{
    printf("RUN CTOR");
    module = this->LoadModel();
    // Set the module to training mode if you want gradients
    module.train();

    std::vector<at::Tensor> parameters;
    for (const auto& params : module.parameters()) {
        parameters.push_back(params);
    }
    //pro celou tridu ne pro kazdy runNN
    optim = new torch::optim::SGD(parameters, /*lr=*/0.1);

}

NEURON_PLUGINPlugin::~NEURON_PLUGINPlugin()
{
    // free(this->optim);
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

int NEURON_PLUGINPlugin::pre_update(Flow& rec, Packet& pkt)
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

// catch 30 packets of first 100 bytes, jak sakra funguje tohle vzbirani flow :D proc jsou vsechnz
// flow #23, ID pluginu? TODO::
int NEURON_PLUGINPlugin::post_update(Flow& rec, const Packet& pkt)
{
    neuronRecord* data = static_cast<neuronRecord*>(rec.get_extension(neuronRecord::REGISTERED_ID));
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
    /// zatim seru an smer
    // printf("incoming packet.len %d \n", pkt.packet_len);
    // printf("incoming payload.len %d \n", pkt.payload_len);

    //  data->packets[data->order].size = MIN(CONTENT_SIZE, pkt.payload_len); // watchout for
    ///  overflow
    data->packets[data->order].size
        = pkt.payload_len < CONTENT_SIZE ? pkt.payload_len : CONTENT_SIZE;
    // printf("size saved on order %d = %d \n", data->order, data->packets[data->order].size);
    printf("\n");

    memcpy(
        data->packets[data->order].data,
        pkt.payload,
        data->packets[data->order].size); // just copy first CONTENT_SIZE packets
}

void NEURON_PLUGINPlugin::pre_export(Flow& rec)
{
    printf("PRE EXPORT:\n");

    auto size_tensor = torch::tensor({});
    neuronRecord* data = static_cast<neuronRecord*>(rec.get_extension(neuronRecord::REGISTERED_ID));


    for (size_t i = 0; i < BUFFER_COUNT; i++) {

        auto size = data->packets[i].size;
        size_tensor = torch::cat({size_tensor, torch::tensor({size})}, 0);
    }
    // std::cout << std::endl << "SIZE TENSOR" << "\t" << size_tensor << std::endl; //todo::dump
    // into bin file
    runNN(size_tensor);
}

void NEURON_PLUGINPlugin::runNN(torch::Tensor input)
{
    printf("RUN NN\n");
    // Create an example input tensor
    torch::Tensor tmp = torch::linspace(1.0, 30.0, 30);
    std::cout << "TMP TENSOR" << "\t" << tmp << std::endl; //todo::dump

    // torch::Tensor tmp = torch::randn({30, 30});




    this->optim->zero_grad();
    for (auto i = 0; i < 10; i++) {
        torch::Tensor loss = module.run_method("training_step", tmp).toTensor();
        loss.backward();
        this->optim->step();
        std::cout << "Epoch: " << i << " | Loss: " << loss.item<float>() << std::endl;
        // printParams(module);
    }
    // printParams(module);
        /// Run inference on the CPU
    torch::Tensor output = module.forward({input}).toTensor();
    std::cout << "Output: " << output << std::endl;

}

void NEURON_PLUGINPlugin::printParams(torch::jit::script::Module model)
{
    // Get the parameters of the module
    std::vector<torch::Tensor> params;
    get_parameters(std::make_shared<torch::jit::script::Module>(model), params);

        std::cout << "Value:\n" << params << std::endl;

}

// Load torchscript model from fileS
torch::jit::script::Module NEURON_PLUGINPlugin::LoadModel()
{
    torch::jit::script::Module loaded_model;
    try {
        loaded_model = torch::jit::load("../tmp/scripted_model.pth");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw e;
    }

    return loaded_model;
}

// TODO might eb worth it for taking params for gradients
void NEURON_PLUGINPlugin::get_parameters(
    std::shared_ptr<torch::jit::script::Module> module,
    std::vector<torch::Tensor>& params)
{
    for (const auto& tensor : module->parameters()) {
        if (tensor.requires_grad())
            params.push_back(tensor);
    }
}


// --------------- NOTES --------------- //


    /// Not that sure what this is..
    ///
    // // Iterate through named parameters and access them
    // for (const auto& pair : module.named_parameters()) {
    //     const std::string& name = pair.name;
    //     torch::Tensor parameter = pair.value;

    //         // Access the gradient tensor
    //     torch::Tensor gradient_tensor = parameter.grad();

    //     if (gradient_tensor.defined()) {
    //         // Print the parameter name and the gradient values
    //         std::cout << "Parameter name: " << name << ", Gradient values: " << gradient_tensor
    //         << std::endl;
    //     } else {
    //         // If gradient is not defined, it's likely a non-trainable parameter
    //         std::cout << "Parameter name: " << name << " is not trainable (no gradient
    //         available)." << std::endl;
    //     }

    // std::cout<< module.parameters() <<std::endl;
    // std::cout<< module->parameters()[0] <<std::endl;
    // std::cout<< module->parameters()[0].grad <<std::endl;
    // print(module.parameters()[0]);
    // print(module.parameters()[0].grad);

    // auto loss_tensor = loss.toTensor();
    // loss_tensor.backward();
    // std::cout << "loss.toTensor(): " << loss_tensor << std::endl;
    
///This is not working
    // // Print the parameters
    // for (const auto& par : params) {
    //     std::string name = par.name();
    //     torch::Tensor value = par.values(); // Assuming the parameter is a tensor

    //     // Print the information
    //     std::cout << "Name: " << name << ", Type: " << value << std::endl;
    //     std::cout << "Value:\n" << value << std::endl;
    // }

} // namespace ipxp
