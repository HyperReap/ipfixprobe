/**
 * \file neuron_plugin.hpp
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

#ifndef IPXP_PROCESS_NEURON_PLUGIN_HPP
#define IPXP_PROCESS_NEURON_PLUGIN_HPP

#include <cstring>

#ifdef WITH_NEMEA
  #include "fields.h"
#endif

#include <ipfixprobe/process.hpp>
#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/packet.hpp>
#include <ipfixprobe/ipfix-elements.hpp>

namespace ipxp {

#define LEARNING_RATE      0.1
#define CONTENT_SIZE       150 //max length of packet
#define BUFFER_COUNT       30 // packets taken from flow
#define EPOCH_COUNT_LIMIT  11 // epoch for training
#define EPOCH_SIZE_LIMIT   128 // flows in epoch
#define BATCH_SIZE         16 // flows in batch

#define NEURON_PLUGIN_UNIREC_TEMPLATE "NEURON_CONTENT" /* TODO: unirec template */

UR_FIELDS (
   /* TODO: unirec fields definition */
   bytes NEURON_CONTENT
)

class NeuralOptParser : public OptionsParser
{
public:
   bool m_inference;
   std::string m_model_path;
   std::string m_state_dict_path;

   NeuralOptParser() : OptionsParser("neural", "Plugin for training/inference using neural networks of packet flows"), m_inference(false), m_model_path("../tests/neuralModels/scripted_model.pth"), m_state_dict_path("../tests/neuralModels/state_dict_values.pt")
   {
      register_option("i", "inference", "", "Setup plugin for inference mode", [this](const char *arg){m_inference = true; return true;}, OptionFlags::NoArgument);
      register_option("m", "model", "", "Neural network model in tochscript", [this](const char *arg){m_model_path = arg; return true;}, OptionFlags::RequiredArgument);
      register_option("s", "state_dict", "", "State_dict of the model", [this](const char *arg){m_state_dict_path = arg; return true;}, OptionFlags::OptionalArgument);
   }
};

struct neuroContentArray {
   neuroContentArray() : size(0.0){ };
   // neuroContentArray() : size(0.0), data(new uint8_t[CONTENT_SIZE + 1]) {}
   // ~neuroContentArray() { delete[] data; }

   float size;
   uint8_t data[CONTENT_SIZE+1];
   // uint8_t* data;
};

/**
 * \brief Flow record extension header for storing parsed NEURON_PLUGIN data.
 */
struct neuronRecord : public RecordExt {
   static int REGISTERED_ID;

   //first 30 packets of single flow
   neuroContentArray packets[BUFFER_COUNT]; 
   
   //order of incoming pakcets, starting at 0
   uint8_t order;

   neuronRecord() : RecordExt(REGISTERED_ID)
   {
      order = 0;
   }

#ifdef WITH_NEMEA
   void fill_unirec(ur_template_t *tmplt, void *record) override
   {
   }

   const char *get_unirec_tmplt() const
   {
      return NEURON_PLUGIN_UNIREC_TEMPLATE;
   }
#endif

   int fill_ipfix(uint8_t *buffer, int size) override
   {
      return 0;
   }

   const char **get_ipfix_tmplt() const
   {
      static const char *ipfix_template[] = {
         IPFIX_NEURON_TEMPLATE(IPFIX_FIELD_NAMES)
         NULL
      };
      return ipfix_template;
   }
};

/**
 * \brief Process plugin for parsing NEURON_PLUGIN packets.
 */
class NEURON_PLUGINPlugin : public ProcessPlugin
{
public:
   NEURON_PLUGINPlugin();
   ~NEURON_PLUGINPlugin();
   void init(const char *params);
   void close();
   OptionsParser *get_parser() const { return new NeuralOptParser(); }
   std::string get_name() const { return "neuron_plugin"; }
   RecordExt *get_ext() const { return new neuronRecord(); }
   ProcessPlugin *copy();

   void update_record(neuronRecord *data, const Packet &pkt);
   int pre_create(Packet &pkt);
   int post_create(Flow &rec, const Packet &pkt);
   int pre_update(Flow &rec, Packet &pkt);
   int post_update(Flow &rec, const Packet &pkt);
   void pre_export(Flow &rec);

   void prepare_data(neuronRecord *data);
   void nn_inference();
   void nn_training();
   void runNN(torch::Tensor tensor);
   torch::jit::script::Module load_model();
   void print_parameters(torch::jit::script::Module model);
   void get_parameters(std::shared_ptr<torch::jit::script::Module> module, std::vector<torch::Tensor>& params);
   void save_state_dict();
   void set_state_dict_parameters(std::vector<torch::Tensor>  loaded_state_dict);
   std::vector<torch::Tensor>  load_state_dict();
   torch::Tensor create_tensor_based_on_flow_array();

   // void set_state_dict_parameters(torch::OrderedDict<std::string, torch::Tensor>  loaded_state_dict);
   // torch::OrderedDict<std::string, torch::Tensor>  load_state_dict();





   private:
   bool is_inference_mode;
   std::string model_path;
   std::string state_dict_path;

   torch::jit::script::Module _model;
   // torch::optim::Adam* _optimizer;
   torch::optim::SGD* _optimizer;

   std::vector<neuronRecord*> _flow_array;

   double _learning_rate;
   unsigned _content_size; // max length of packet
   unsigned _buffer_count; // packets taken from flow
   unsigned _epoch_count; // epoch for training
   unsigned _epoch_count_limit; // epoch for training
   unsigned _epoch_size; // flows in epoch
   unsigned _epoch_size_limit; // maximum flows in epoch
   unsigned _batch_size; // flows in batch

};

}
#endif /* IPXP_PROCESS_NEURON_PLUGIN_HPP */

