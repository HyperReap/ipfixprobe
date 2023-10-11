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

#define CONTENT_SIZE       100
#define BUFFER_COUNT       30

#define NEURON_PLUGIN_UNIREC_TEMPLATE "NEURO_CONTENT" /* TODO: unirec template */

UR_FIELDS (
   /* TODO: unirec fields definition */
   bytes NEURO_CONTENT
)

struct neuroContentArray {
   neuroContentArray() : size(0){ };
   uint8_t size;
   uint8_t data[CONTENT_SIZE];
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
         IPFIX_NEURON_PLUGIN_TEMPLATE(IPFIX_FIELD_NAMES)
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
   OptionsParser *get_parser() const { return new OptionsParser("neuron_plugin", "Parse NEURON_PLUGIN traffic"); }
   std::string get_name() const { return "neuron_plugin"; }
   RecordExt *get_ext() const { return new neuronRecord(); }
   ProcessPlugin *copy();

   int pre_create(Packet &pkt);
   int post_create(Flow &rec, const Packet &pkt);
   int pre_update(Flow &rec, Packet &pkt);
   int post_update(Flow &rec, const Packet &pkt);
   void pre_export(Flow &rec);
};

}
#endif /* IPXP_PROCESS_NEURON_PLUGIN_HPP */

