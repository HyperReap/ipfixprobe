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

    _learning_rate = LEARNING_RATE;
    _content_size = CONTENT_SIZE;           // max length of packet
    _buffer_count = BUFFER_COUNT;           // packets taken from flow
    _batch_size = BATCH_SIZE;               // flows in batch
    _epoch_count_limit = EPOCH_COUNT_LIMIT; // epoch for training
    _epoch_size_limit = EPOCH_SIZE_LIMIT;   // flows in epoch

    _epoch_count = 0;       
    _epoch_size = 0; 


    this->_model = this->LoadModel();
    // Set the module to training mode if you want gradients
    this->_model.train();

    std::vector<at::Tensor> parameters;
    for (const auto& params : _model.parameters()) {
        parameters.push_back(params);
    }

    this->_optimizer = new torch::optim::SGD(parameters, _learning_rate);

}

NEURON_PLUGINPlugin::~NEURON_PLUGINPlugin()
{
    if(_flow_array.size()>0)
        _flow_array.clear();

    // free(this->_optimizer); //TODO:: this calls second free?
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
    // std::cout<<"inference"<<std::endl;
    // nn_inference();
    return 0;
}



int NEURON_PLUGINPlugin::post_create(Flow& rec, const Packet& pkt)
{
    neuronRecord* record = new neuronRecord(); // created new flow, can keep 30 packets from it
    rec.add_extension(record); // register this flow
    update_record(record, pkt);

    return 0;
}

// catch 30 packets of first 100 bytes, jak sakra funguje tohle vzbirani flow :D proc jsou vsechnz
int NEURON_PLUGINPlugin::post_update(Flow& rec, const Packet& pkt)
{
    neuronRecord* data = static_cast<neuronRecord*>(rec.get_extension(neuronRecord::REGISTERED_ID));
    data->order++;

    if (data->order > BUFFER_COUNT) {
        // printf("Too many packets in this flow > %d\n", BUFFER_COUNT);
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
    if (pkt.packet_len_wire < CONTENT_SIZE)
    {
        data->packets[data->order].size = pkt.packet_len_wire;
    }
    else 
    {
        data->packets[data->order].size = CONTENT_SIZE;
    }

    // std::cout<< "order :" << (int)data->order << " | packet len wire: " << pkt.packet_len_wire << std::endl;

    // memcpy(
    //     data->packets[data->order].data,
    //     pkt.packet,
    //     data->packets[data->order].size); // just copy first CONTENT_SIZE packets
}

void NEURON_PLUGINPlugin::pre_export(Flow& rec)
{
    // printf("PRE EXPORT:\n");
    if(this->_epoch_count > this->_epoch_count_limit) 
    {
        printf("End of EPOCHS - Save Model\n");
        //https://discuss.pytorch.org/t/saving-and-loading-model-with-libtorch-c/184482
        std::vector<torch::Tensor> params;
        get_parameters(std::make_shared<torch::jit::script::Module>(_model), params);
         // Save the tensor to a file
        torch::save(params, "../models/params.pt");

        printParams(_model);

        exit(1);
    }

    // add record to flowArray
    neuronRecord* data = static_cast<neuronRecord*>(rec.get_extension(neuronRecord::REGISTERED_ID));
    _flow_array.push_back(data);

    // check if there is enough flows in batch
    if(_flow_array.size() < _batch_size)
    {
        // std::cout << "Not enough flows. Current Count: " << _flow_array.size() << std::endl;
        return;
    }

    this->_epoch_size += this->_flow_array.size();


    // auto tmp = this->_model.state_dict();
    std::cout<<"training- epochSize:"<< _epoch_size <<std::endl;
    nn_training();


    printf("clear flow array\n");
    this->_flow_array.clear();

    if(this->_epoch_size >= this->_epoch_size_limit)
    {
        this->_epoch_size = 0;
        this->_epoch_count++;
    }

    // auto size_tensor = torch::tensor({});

    // for (size_t i = 0; i < BUFFER_COUNT; i++) {

    //     auto size = data->packets[i].size;
    //     size_tensor = torch::cat({size_tensor, torch::tensor({size})}, 0);
    // }
    // // std::cout << std::endl << "SIZE TENSOR" << "\t" << size_tensor << std::endl; //todo::dump
    // // into bin file
    // runNN(size_tensor);
}

void NEURON_PLUGINPlugin::nn_training()
{
    int batch_count_in_epoch = 0;
    torch::Tensor concatenated_tensor = torch::empty({_batch_size, _buffer_count}, torch::kFloat32);

    for (auto record : this->_flow_array)
    {
        torch::Tensor packet_lengths_tensor = torch::empty({_buffer_count}, torch::kFloat32);

        for (size_t i = 0; i < BUFFER_COUNT; i++) 
        {
            float size = record->packets[i].size; /// todo into float
            // if(size <= 0)
            //     size = 99;
            packet_lengths_tensor[i] = (size);
        }

        concatenated_tensor[batch_count_in_epoch] = packet_lengths_tensor;


        batch_count_in_epoch++;
    }
        
        std::cout<<"batch tensor: "<<concatenated_tensor<<std::endl;

        // torch::Tensor tmp = torch::randn({_batch_size, _buffer_count});

        this->_optimizer->zero_grad();
        torch::Tensor loss = this->_model.run_method("training_step", concatenated_tensor).toTensor();

        loss.backward();
        this->_optimizer->step();
        
        // std::cout << "Epoch: " << this->_epoch_count << " | batch_count_in_epoch: " << batch_count_in_epoch << " | Loss: " << loss.item<float>() << std::endl;
        std::cout << "Epoch: " << this->_epoch_count << " | Loss: " << loss.item<float>() << std::endl;


}

void NEURON_PLUGINPlugin::nn_inference()
{
    // Load the saved tensor back
    torch::Tensor loaded_params;
    torch::load(loaded_params, "../models/params.pt");
    //TODO load as ordered dict, same way as get_params work.

    // _model.load_state_dict(loaded_params);
    // std::vector<torch::Tensor> params;
    // get_parameters(std::make_shared<torch::jit::script::Module>(_model), params);

    //  // Load the trained parameters into the loaded model
    // for (size_t i = 0; i < params.size(); ++i) {
    //     params[i].set_data(loaded_params[i]);
    // }

    printf("run inference on saved model\n");
    torch::Tensor tmp = torch::randn({30, 30});
    auto output = _model.forward({tmp}).toTensor();

    return;
}


// TODO add data as parameter
    //to count number of epochs
void NEURON_PLUGINPlugin::runNN(torch::Tensor input)
{
    printf("RUN NN\n");

    for (auto i = 0; i < 50; i++) {
    printf("batch_count_in_epoch\n");

        this->_optimizer->zero_grad();
        torch::Tensor tmp = torch::randn({30, 30});

        torch::Tensor loss = this->_model.run_method("training_step", tmp).toTensor();
        
        loss.backward();
        this->_optimizer->step();
        
        std::cout << "Epoch: " << i << " | Loss: " << loss.item<float>() << std::endl;
        // printParams(_model);
    }


    torch::Tensor ln = torch::linspace(1.0, 30.0, 30);
    torch::Tensor output = _model.forward({ln}).toTensor();
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
        loaded_model = torch::jit::load("../models/scripted_model.pth");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw e;
    }

    return loaded_model;
}

//  https://discuss.pytorch.org/t/libtorch-save-mnist-c-examples-trained-model-into-a-file-and-load-in-from-another-c-file-to-use-for-prediction/51681/11
// These are scenarios that C++/TorchScript serialization doesn’t support:
//      Save as C++ model using torch::save, load using torch::jit::load in C++
//      Save as C++ model using torch::save, load using torch.load in Python
//      Save as C++ model using torch::save, load using torch.jit.load in Python

// As such trying to save models parameters and load them into model



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
