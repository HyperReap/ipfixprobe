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

#define LEARNING_RATE      0.01
#define CONTENT_SIZE       3 //max length of packet
#define BUFFER_COUNT       1 // packets taken from flow
#define EPOCH_COUNT_LIMIT  4 // epoch for training
#define EPOCH_SIZE_LIMIT   1024 // flows in epoch
#define BATCH_SIZE         64 // flows in batch
#define DEFAULT_STATE_DICT "~/ipfixprobe/tests/neuralModels/state_dict_values.pt"

#define NEURON_PLUGIN_UNIREC_TEMPLATE "NEURON_CONTENT" /* TODO: unirec template */

UR_FIELDS (
   /* TODO: unirec fields definition */
   bytes NEURON_CONTENT
)

class NeuralOptParser : public OptionsParser
{
public:
   bool m_inference;
   bool m_continue;
   bool m_dump;
   float m_lr;
   std::string m_model_path;
   std::string m_state_dict_path;
   std::string m_dump_path;

   NeuralOptParser() : OptionsParser("neural", "Plugin for training/inference using neural networks of packet flows"),
    m_inference(false), m_model_path("../tests/neuralModels/scripted_model.pth"),
    m_dump_path("../tests/neuralModels/tensors.txt"), m_dump(false),
    m_state_dict_path(DEFAULT_STATE_DICT), m_continue(false), m_lr(LEARNING_RATE)
   {
      register_option("i", "inference", "", "Setup plugin for inference mode", [this](const char *arg){m_inference = true; return true;}, OptionFlags::NoArgument);
      register_option("m", "model", "", "Neural network model in tochscript", [this](const char *arg){m_model_path = arg; return true;}, OptionFlags::RequiredArgument);
      register_option("d", "dump", "", "Collect dataset as tensors to specified path", [this](const char *arg){m_dump = true; m_dump_path = arg; return true;}, OptionFlags::OptionalArgument);
      register_option("s", "state_dict", "", "State_dict of the model", [this](const char *arg){m_state_dict_path = arg; return true;}, OptionFlags::OptionalArgument);
      register_option("c", "continue", "", "Continue training with specified State_dict of the model", [this](const char *arg){m_continue = true; return true;}, OptionFlags::OptionalArgument);
      register_option("lr", "lr", "", "set learnign rate for training", [this](const char *arg){m_lr = std::stof(arg); return true;}, OptionFlags::OptionalArgument);
   }
};

struct neuroContentArray {
   neuroContentArray() : size(0.0){ };

   size_t size;

   uint8_t data[CONTENT_SIZE+1];
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

   void dump_data();
   void nn_inference();
   void nn_training();
   torch::jit::script::Module load_model();
   void save_state_dict();
   void set_state_dict_parameters(std::vector<torch::Tensor>  loaded_state_dict);
   std::vector<torch::Tensor>  load_state_dict();
   torch::Tensor create_tensor_based_on_flow_array();

   private:
   bool is_inference_mode;
   bool should_continue_training;
   bool should_dump_tensors;
   std::string dump_path;
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

   bool _should_skip_rest_of_traffic; // skip the rest when the model is trained

};

}
#endif /* IPXP_PROCESS_NEURON_PLUGIN_HPP */

