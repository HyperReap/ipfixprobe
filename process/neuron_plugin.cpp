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
#include <torch/torch.h>

#include "neuron_plugin.hpp"
using namespace torch::autograd;

namespace ipxp {

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int neuronRecord::REGISTERED_ID = -1;

__attribute__((constructor)) static void register_this_plugin()
{
   static PluginRecord rec = PluginRecord("neuron", [](){return new NEURON_PLUGINPlugin();});
   register_plugin(&rec);
   neuronRecord::REGISTERED_ID = register_extension();
}

NEURON_PLUGINPlugin::NEURON_PLUGINPlugin()
{
}

NEURON_PLUGINPlugin::~NEURON_PLUGINPlugin()
{
   close();
}

void NEURON_PLUGINPlugin::init(const char *params)
{
}

void NEURON_PLUGINPlugin::close()
{
}


void NEURON_PLUGINPlugin::update_record(neuronRecord *data, const Packet &pkt)
{
   //zatim seru an smer
   printf("incoming %d \n", pkt.packet_len);
   data->packets[data->order].size = MIN(CONTENT_SIZE, pkt.payload_len); //watchout for overflow
   memcpy(data->packets[data->order].data, pkt.payload, data->packets[data->order].size); //just copy first CONTENT_SIZE packets 
}

ProcessPlugin *NEURON_PLUGINPlugin::copy()
{
   return new NEURON_PLUGINPlugin(*this);
}

int NEURON_PLUGINPlugin::pre_create(Packet &pkt)
{
   return 0;
}

int NEURON_PLUGINPlugin::post_create(Flow &rec, const Packet &pkt)
{
   neuronRecord *record = new neuronRecord(); //created new flow, can keep 30 packets from it

   rec.add_extension(record); //register this flow
   update_record(record, pkt);
   
   return 0;
}

int NEURON_PLUGINPlugin::pre_update(Flow &rec, Packet &pkt)
{
   return 0;
}

int NEURON_PLUGINPlugin::post_update(Flow &rec, const Packet &pkt)
{
   neuronRecord *data = static_cast<neuronRecord *>(rec.get_extension(neuronRecord::REGISTERED_ID));
   data->order++;
   if(data->order > BUFFER_COUNT)
      return 0;

   update_record(data, pkt);

   return 0;
}

void NEURON_PLUGINPlugin::pre_export(Flow &rec)
{
   printf("EXPORT:\n");

   auto size_tensor = torch::tensor({0});
   auto empty_tensor = torch::empty({0});
   neuronRecord *data = static_cast<neuronRecord *>(rec.get_extension(neuronRecord::REGISTERED_ID));
   

   for (size_t i = 0; i < BUFFER_COUNT; i++)
   {
      auto size = data->packets[i].size;
      printf("size: %d \n", size);
      size_tensor = torch::cat({size_tensor, torch::tensor({size})},0);
      /* code */
   }

   std::cout << std::endl << "SIZE TENSOR" << std::endl << size_tensor << std::endl;

}
}

