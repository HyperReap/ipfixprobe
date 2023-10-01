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

#include "neuron_plugin.hpp"

namespace ipxp {

int RecordExtNEURON_PLUGIN::REGISTERED_ID = -1;

__attribute__((constructor)) static void register_this_plugin()
{
   static PluginRecord rec = PluginRecord("neuron_plugin", [](){return new NEURON_PLUGINPlugin();});
   register_plugin(&rec);
   RecordExtNEURON_PLUGIN::REGISTERED_ID = register_extension();
}

NEURON_PLUGINPlugin::NEURON_PLUGINPlugin()
{
}

NEURON_PLUGINPlugin::~NEURON_PLUGINPlugin()
{
}

void NEURON_PLUGINPlugin::init(const char *params)
{
}

void NEURON_PLUGINPlugin::close()
{
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
   return 0;
}

int NEURON_PLUGINPlugin::pre_update(Flow &rec, Packet &pkt)
{
   return 0;
}

int NEURON_PLUGINPlugin::post_update(Flow &rec, const Packet &pkt)
{
   return 0;
}

void NEURON_PLUGINPlugin::pre_export(Flow &rec)
{
}

}

