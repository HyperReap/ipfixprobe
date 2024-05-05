/**
 * \file neural.cpp
 * \brief Plugin for parsing neura; traffic.
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

#include "neural.hpp"
using namespace torch::autograd;

namespace ipxp {

int NeuralRecord::REGISTERED_ID = -1;

__attribute__((constructor)) static void register_this_plugin()
{
    static PluginRecord rec = PluginRecord("neural", []() { return new NEURALPlugin(); });
    register_plugin(&rec);
    NeuralRecord::REGISTERED_ID = register_extension();
}

NEURALPlugin::NEURALPlugin() 
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
    
    model_path = "../tests/neuralModels/scripted_model.pth";
    state_dict_path = DEFAULT_STATE_DICT;
    should_continue_training = false;
    should_dump_tensors = false;
}

NEURALPlugin::~NEURALPlugin()
{
    if(_flow_array.size() > 0)
        _flow_array.clear();

    // free(this->_optimizer); //this calls second free, keeping here as comment for future
    close();
}

void NEURALPlugin::init(const char* params) 
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
    this->should_continue_training = parser.m_continue;
    this->should_dump_tensors = parser.m_dump;
    this->dump_path = parser.m_dump_path;
    this->_learning_rate = parser.m_lr;
    

    std::cout<<"inference:" << this->is_inference_mode<<std::endl;
    std::cout<<"model_path:" << this->model_path<<std::endl;
    std::cout<<"state_dict:" << this->state_dict_path<<std::endl<<std::endl;
    std::cout<<"continue:" << this->should_continue_training<<std::endl<<std::endl;
    std::cout<<"dump:" << this->should_dump_tensors<<std::endl<<std::endl;
    std::cout<<"dump path:" << this->dump_path<<std::endl<<std::endl;
    std::cout<<"lr:" << this->_learning_rate<<std::endl<<std::endl;

    this->_model = this->load_model();

    _should_skip_rest_of_traffic  = false;

    if(this->is_inference_mode)
    {
        this->_batch_size = 1;
        std::vector<torch::Tensor> loaded_params = load_state_dict();
        set_state_dict_parameters(loaded_params);
    }
    else
    {
        // Set the module to training mode if you want gradients
        this->_model.train();

        if(this->should_continue_training) 
        {
            std::vector<torch::Tensor> loaded_params = load_state_dict();
            if(loaded_params.size() > 0)
                set_state_dict_parameters(loaded_params);
        }

        std::vector<at::Tensor> parameters;
        for (const auto& params : _model.parameters()) 
        {
            parameters.push_back(params);
        }

        // this->_optimizer = new torch::optim::Adam(parameters, _learning_rate);
        this->_optimizer = new torch::optim::SGD(parameters, _learning_rate);
    }
}

void NEURALPlugin::close() {}

ProcessPlugin* NEURALPlugin::copy()
{
    return new NEURALPlugin(*this);
}

int NEURALPlugin::pre_create(Packet& pkt)
{
    return 0;
}

int NEURALPlugin::pre_update(Flow& rec, Packet& pkt)
{
    return 0;
}



int NEURALPlugin::post_create(Flow& rec, const Packet& pkt)
{
    NeuralRecord* record = new NeuralRecord(); // created new flow, can keep 30 packets from it
    rec.add_extension(record); // register this flow
    update_record(record, pkt);

    return 0;
}


int NEURALPlugin::post_update(Flow& rec, const Packet& pkt)
{
    NeuralRecord* data = static_cast<NeuralRecord*>(rec.get_extension(NeuralRecord::REGISTERED_ID));
    update_record(data, pkt);
    return 0;
}

void NEURALPlugin::update_record(NeuralRecord* data, const Packet& pkt)
{
    if(_should_skip_rest_of_traffic)
        return;

    if(data->order >= this->_buffer_count)
    {
        // printf("Too many packets in this flow > %d\n", BUFFER_COUNT);
        return;
    }

    ///  overflow
    size_t min = pkt.payload_len < CONTENT_SIZE ? pkt.payload_len: CONTENT_SIZE;
    data->packets[data->order].size = min;

    // std::cout<< "order :" << (int)data->order << " | packet len : " << pkt.packet_len << std::endl;

    if(data->order > BUFFER_COUNT)
    {
        std::cerr << "somehow got wrong order number:" << (int)data->order <<std::endl;
    }

    auto& target_packet = data->packets[data->order];
    std::copy(pkt.payload, pkt.payload + target_packet.size , target_packet.data);

    data->order++;
}


void NEURALPlugin::pre_export(Flow& rec)
{
    if(_should_skip_rest_of_traffic)
        return;

    // add record to flowArray
    NeuralRecord* data = static_cast<NeuralRecord*>(rec.get_extension(NeuralRecord::REGISTERED_ID));
    _flow_array.push_back(data);

    // check if there is enough flows in batch
    if(_flow_array.size() < _batch_size)
    {
        // std::cout << "Not enough flows. Current Count: " << _flow_array.size() << std::endl;
        return;
    }

    if(this->should_dump_tensors)
    {
        dump_data();
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
            _should_skip_rest_of_traffic  = true;
        }

        if(!_should_skip_rest_of_traffic)
        {
            this->_epoch_size += this->_flow_array.size();

            std::cout<<"training - epochSize: "<< _epoch_size <<std::endl;
            nn_training();

            if(this->_epoch_size >= this->_epoch_size_limit)
            {
                this->_epoch_size = 0;
                this->_epoch_count++;
            }
        }
    }

    // printf("clear flow array\n");
    this->_flow_array.clear();
}

void NEURALPlugin::nn_training()
{
    auto tensor  = create_tensor_based_on_flow_array();

    // std::cout<<"tensor: "<<std::endl<<tensor<<std::endl;
    // auto mean = torch::mean(tensor);
    // std::cout<<"mean: "<<mean<<std::endl;

    this->_optimizer->zero_grad();
    torch::Tensor loss = this->_model.run_method("training_step", tensor).toTensor();

    loss.backward();
    this->_optimizer->step();
    
    std::cout << "Epoch: " << this->_epoch_count << " | Loss: " << loss.item<float>() << std::endl;
}

void NEURALPlugin::nn_inference()
{
    // printf("run inference on saved model\n");
    auto tensor  = create_tensor_based_on_flow_array();
    //todo single Flow -> batchsize = 1
    // tady ukoncoit casovac nad jendim tokem, apak vzyhodnoti Cas
    //execution time
    //kolik Flows jsem SCHPNY UDELAT ZA SEUNDU
    //PRI NEJHORSIm tam dat ciste timewatch

    

    // std::cout<<"tensor: "<<std::endl<<tensor<<std::endl;
    // auto mean = torch::mean(tensor);
    // std::cout<<"mean: "<<mean<<std::endl;

    auto output = _model.forward({tensor}).toTensor()*255;

    std::cout<<output<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;
    
    return;
}

torch::Tensor NEURALPlugin::create_tensor_based_on_flow_array()
{
    int batches_in_epoch = 0; 
    // return torch::randn({_batch_size, _buffer_count, _content_size}, torch::kFloat32);
    torch::Tensor concatenated_tensor = torch::zeros({_batch_size, _buffer_count, _content_size}, torch::kFloat32);

    for (auto record : this->_flow_array)
    {
        torch::Tensor packet_data_tensor = torch::zeros({_buffer_count, _content_size}, torch::kFloat32);

        for (size_t i = 0; i < _buffer_count; i++) 
        {
            size_t size = record->packets[i].size;
            for (size_t j = 0; j < size; j++)
            {
                auto data = record->packets[i].data[j];
                packet_data_tensor[i][j] = data/255.0; 
            }
        }
        // Concatenate packet_data_tensor along the batch dimension
        concatenated_tensor[batches_in_epoch] = packet_data_tensor;
        batches_in_epoch++;
    }
    return concatenated_tensor;
}


void NEURALPlugin::dump_data()
{
    std::ofstream outfile(this->dump_path, std::ios_base::app);


// Read File for t
    // std::ifstream infile(this->dump_path);

    // if (!infile.is_open()) {
    //     std::cerr << "Failed to open file for read." << std::endl;
    //     return;
    // }

    // while (!infile.eof()) 
    // {
    //     // Load tensor from file
    //     torch::Tensor tensor;
    //     torch::load(tensor, infile);
    //     std::cout<<tensor<<std::endl;
    //     // Check if tensor is non-empty
    //     if (!tensor.defined()) {
    //         break; // Reached end of file or error occurred
    //     }
    // }

    // exit(1);

    // Check if the file is opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

      // Convert each tensor to a string
    auto tensor  = create_tensor_based_on_flow_array();
    torch::save(tensor, outfile);
   
    outfile.close();
}

//https://discuss.pytorch.org/t/saving-and-loading-model-with-libtorch-c/184482
void NEURALPlugin::save_state_dict()
{
    printf("End of EPOCHS - Save Model\n");
    auto named_parameters = _model.named_parameters();
    std::vector<std::string> names; //hypothetically dont need those, will always be in the same order as it should be
    std::vector<torch::Tensor> tensors;

 
    std::cout << "state_dict:" << std::endl;
    for (const auto& named_param : named_parameters) 
    {
        auto name = named_param.name;
        auto tensor = named_param.value;

        names.push_back(name);
        tensors.push_back(tensor);
    }
        std::cout << names << std::endl ;
        std::cout << tensors << std::endl ;
    
    try 
    {
        torch::save(tensors, state_dict_path);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error saving the state dictionary\n" << e.what() <<std::endl;
        return;
    }
}

std::vector<torch::Tensor> NEURALPlugin::load_state_dict()
{
    std::vector<torch::Tensor> loaded_state_dict;
    try 
    {
        torch::load(loaded_state_dict, state_dict_path);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error loading the state dictionary\n" <<std::endl;
    }
    return loaded_state_dict;
}

//  https://discuss.pytorch.org/t/libtorch-save-mnist-c-examples-trained-model-into-a-file-and-load-in-from-another-c-file-to-use-for-prediction/51681/11
// These are scenarios that C++/TorchScript serialization doesn’t support:
//      Save as C++ model using torch::save, load using torch::jit::load in C++
//      Save as C++ model using torch::save, load using torch.load in Python
//      Save as C++ model using torch::save, load using torch.jit.load in Python

// As such trying to save models parameters and load them into model

void NEURALPlugin::set_state_dict_parameters(std::vector<torch::Tensor> loaded_state_dict)
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
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error setting the loaded state dictionary to the model\n";
        std::cerr << e.what();
        return;
    }
}

// Load torchscript model from fileS
torch::jit::script::Module NEURALPlugin::load_model()
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
} // namespace ipxp
