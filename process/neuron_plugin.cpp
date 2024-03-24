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
    _learning_rate = LEARNING_RATE;
    _content_size = CONTENT_SIZE;           // max length of packet
    _buffer_count = BUFFER_COUNT;           // packets taken from flow
    _batch_size = BATCH_SIZE;               // flows in batch
    _epoch_count_limit = EPOCH_COUNT_LIMIT; // epoch for training
    _epoch_size_limit = EPOCH_SIZE_LIMIT;   // flows in epoch

    _epoch_count = 0;       
    _epoch_size = 0; 

    is_inference_mode = false;
    
    model_path = "../tests/neuralModels/scripted_model.pt";
    state_dict_path = "../tests/neuralModels/state_dict_values.pt";
    
}

NEURON_PLUGINPlugin::~NEURON_PLUGINPlugin()
{
    if(_flow_array.size() > 0)
        _flow_array.clear();

    // free(this->_optimizer); //TODO:: this calls second free?
    close();
}

void NEURON_PLUGINPlugin::init(const char* params) 
{
    printf("INIT\n");

    NeuralOptParser parser;
   try {
      parser.parse(params);
   } catch (ParserError &e) {
      throw PluginError(e.what());
   }

    this->is_inference_mode = parser.m_inference;
    this->model_path = parser.m_model_path;
    this->state_dict_path = parser.m_state_dict_path;
    

    std::cout<<"inference:" << this->is_inference_mode<<std::endl;
    std::cout<<"model_path:" << this->model_path<<std::endl;
    std::cout<<"state_dict:" << this->state_dict_path<<std::endl<<std::endl;

    this->_model = this->load_model();

    if(this->is_inference_mode)
    {
        std::vector<torch::Tensor> loaded_params = load_state_dict();
        set_state_dict_parameters(loaded_params);
    }
    else
    {
        // Set the module to training mode if you want gradients
        this->_model.train();

        std::vector<at::Tensor> parameters;
        for (const auto& params : _model.parameters()) {
            parameters.push_back(params);
        }

        // this->_optimizer = new torch::optim::Adam(parameters, _learning_rate);
        this->_optimizer = new torch::optim::SGD(parameters, _learning_rate);
    }
}

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
        data->packets[data->order].size = pkt.packet_len_wire;

    // if (pkt.packet_len_wire < CONTENT_SIZE)
    // {
    //     data->packets[data->order].size = pkt.packet_len_wire;
    // }
    // else 
    // {
    //     data->packets[data->order].size = CONTENT_SIZE;
    // }

    // std::cout<< "order :" << (int)data->order << " | packet len wire: " << pkt.packet_len_wire << std::endl;

    // memcpy(
    //     data->packets[data->order].data,
    //     pkt.packet,
    //     data->packets[data->order].size); // just copy first CONTENT_SIZE packets

    prepare_data(data);
}

void NEURON_PLUGINPlugin::prepare_data(neuronRecord* data)
{
    //TODO SANITIZE DATA or anything we will want
}



void NEURON_PLUGINPlugin::pre_export(Flow& rec)
{
    // printf("PRE EXPORT:\n");
    // add record to flowArray
    neuronRecord* data = static_cast<neuronRecord*>(rec.get_extension(neuronRecord::REGISTERED_ID));
    _flow_array.push_back(data);

    // check if there is enough flows in batch
    if(_flow_array.size() < _batch_size)
    {
        // std::cout << "Not enough flows. Current Count: " << _flow_array.size() << std::endl;
        return;
    }

    if(this->is_inference_mode)
    {
        nn_inference();
    }
    else
    {
        if(this->_epoch_count > this->_epoch_count_limit) 
        {
            save_state_dict();
            exit(1);
        }

        this->_epoch_size += this->_flow_array.size();

        // auto tmp = this->_model.state_dict();
        std::cout<<"training - epochSize: "<< _epoch_size <<std::endl;
        nn_training();

        if(this->_epoch_size >= this->_epoch_size_limit)
        {
            this->_epoch_size = 0;
            this->_epoch_count++;
        }
    }

    printf("clear flow array\n");
    this->_flow_array.clear();
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
        auto tensor  = create_tensor_based_on_flow_array();
        
        // torch::Tensor tmp = torch::rand({_batch_size, _buffer_count});
        // torch::Tensor tmp2 = tmp * (100 - 0) + 0;

        // std::cout<<"batch tensor: "<<concatenated_tensor<<std::endl;

        // std::cout<<"tmp tensor: "<<tmp<<std::endl <<std::endl;

        // seems like normalization fo data is needed:
        // min-max normalization
        // min_value = 0
        // max_value = 200
        // normalized_data = (batch - min_value) / (max_value - min_value)


        this->_optimizer->zero_grad();
        torch::Tensor loss = this->_model.run_method("training_step", tensor).toTensor();

        loss.backward();
        this->_optimizer->step();
        
        // std::cout << "Epoch: " << this->_epoch_count << " | batch_count_in_epoch: " << batch_count_in_epoch << " | Loss: " << loss.item<float>() << std::endl;
        std::cout << "Epoch: " << this->_epoch_count << " | Loss: " << loss.item<float>() << std::endl;


}

void NEURON_PLUGINPlugin::nn_inference()
{
    // Load the saved tensor back
    // std::vector<torch::Tensor> loaded_params = load_state_dict();
    // set_state_dict_parameters(loaded_params);

    //TODO load as ordered dict, same way as get_params work.
    //DIDNT work for ordered dict so used std vector for valeus only, order of values should be same for the same model 

    // _model.load_state_dict(loaded_params);
    // std::vector<torch::Tensor> params;
    // get_parameters(std::make_shared<torch::jit::script::Module>(_model), params);

    //  // Load the trained parameters into the loaded model
    // for (size_t i = 0; i < params.size(); ++i) {
    //     params[i].set_data(loaded_params[i]);
    // }

    printf("run inference on saved model\n");
    // torch::Tensor ln = torch::linspace(1.0, 30.0, 30);
    // auto output = _model.forward({ln}).toTensor();

    
    auto tensor  = create_tensor_based_on_flow_array();

    auto output = _model.forward({tensor}).toTensor();

    std::cout<<output<<std::endl;
    return;
}

torch::Tensor NEURON_PLUGINPlugin::create_tensor_based_on_flow_array()
{
    int batch_count_in_epoch = 0; //todo renaqme, feels weird
    torch::Tensor concatenated_tensor = torch::empty({_batch_size, _buffer_count}, torch::kFloat32);

    for (auto record : this->_flow_array)
    {
        torch::Tensor packet_lengths_tensor = torch::empty({_buffer_count}, torch::kFloat32);

        for (size_t i = 0; i < _buffer_count; i++) 
        {
            float size = record->packets[i].size; 
            packet_lengths_tensor[i] = (size);
        }

        concatenated_tensor[batch_count_in_epoch] = packet_lengths_tensor;


        batch_count_in_epoch++;
    }
    return concatenated_tensor;
}



// TODO remove later on, after we are sure it works as expected, dotn want to lose knowledge now
    //to count number of epochs
void NEURON_PLUGINPlugin::runNN(torch::Tensor input)
{
    printf("RUN NN\n");

    for (auto i = 0; i < 50; i++) {

        this->_optimizer->zero_grad();
        torch::Tensor tmp = torch::randn({30, 30});

        torch::Tensor loss = this->_model.run_method("training_step", tmp).toTensor();
        
        loss.backward();
        this->_optimizer->step();
        
        std::cout << "Epoch: " << i << " | Loss: " << loss.item<float>() << std::endl;
        // print_parameters(_model);
    }


    torch::Tensor ln = torch::linspace(1.0, 30.0, 30);
    torch::Tensor output = _model.forward({ln}).toTensor();
    std::cout << "Output: " << output << std::endl;
    
 
}



        //https://discuss.pytorch.org/t/saving-and-loading-model-with-libtorch-c/184482
void NEURON_PLUGINPlugin::save_state_dict()
{
    printf("End of EPOCHS - Save Model\n");
    auto named_parameters = _model.named_parameters();
    std::vector<std::string> names; //hypothetically dont need those, will always be in the same order as it should be
    std::vector<torch::Tensor> tensors;

    // torch::OrderedDict<std::string, torch::Tensor> state_dict;
 
    std::cout << "state_dict:" << std::endl;
    // get_parameters(std::make_shared<torch::jit::script::Module>(_model), tensors);
    for (const auto& named_param : named_parameters) 
    {
        auto name = named_param.name;
        auto tensor = named_param.value;
        // // // state_dict.insert(name, tensor);

        names.push_back(name);
        tensors.push_back(tensor);
    }
        std::cout << names << std::endl ;
        std::cout << tensors << std::endl ;
    
    try 
    {
        // torch::save(names, "state_dict_names.pt");
        torch::save(tensors, state_dict_path);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error saving the state dictionary\n";
        return;
    }
}

// torch::OrderedDict<std::string, torch::Tensor> NEURON_PLUGINPlugin::load_state_dict()
std::vector<torch::Tensor> NEURON_PLUGINPlugin::load_state_dict()
{
    // torch::OrderedDict<std::string, torch::Tensor> loaded_state_dict;
    std::vector<torch::Tensor> loaded_state_dict;
    try 
    {
        torch::load(loaded_state_dict, state_dict_path);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error loading the state dictionary\n";
    }
    return loaded_state_dict;
}

//  https://discuss.pytorch.org/t/libtorch-save-mnist-c-examples-trained-model-into-a-file-and-load-in-from-another-c-file-to-use-for-prediction/51681/11
// These are scenarios that C++/TorchScript serialization doesn’t support:
//      Save as C++ model using torch::save, load using torch::jit::load in C++
//      Save as C++ model using torch::save, load using torch.load in Python
//      Save as C++ model using torch::save, load using torch.jit.load in Python

// As such trying to save models parameters and load them into model

void NEURON_PLUGINPlugin::set_state_dict_parameters(std::vector<torch::Tensor> loaded_state_dict)
{
    try 
    {
        torch::NoGradGuard no_grad;
        auto named_parameters = _model.named_parameters();

        int iteration = 0;
        for (const auto& named_param : named_parameters) 
        {   
            torch::Tensor param = loaded_state_dict.at(iteration);
            std::cout<< "setting parameter: " << named_param.name << "to value:" 
                <<std::endl << param <<std::endl;
            named_param.value.set_(param);
            iteration++;
        }        

        // Iterate through the named parameters and set them
        // for (const auto& named_param : named_parameters) 
        // {
        //     const std::string& name = named_param.name;
        //     if (loaded_state_dict.contains(name)) {
        //         named_param.value.set_(loaded_state_dict[name]);
        //     }
        // }


    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error setting the loaded state dictionary to the model\n";
        std::cerr << e.what();
        return;
    }
}

// Load torchscript model from fileS
torch::jit::script::Module NEURON_PLUGINPlugin::load_model()
{
    torch::jit::script::Module loaded_model;
    try {
        loaded_model = torch::jit::load(this->model_path);
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

void NEURON_PLUGINPlugin::print_parameters(torch::jit::script::Module model)
{
    // Get the parameters of the module
    std::vector<torch::Tensor> params;
    get_parameters(std::make_shared<torch::jit::script::Module>(model), params);

        std::cout << "Value:\n" << params << std::endl;

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
