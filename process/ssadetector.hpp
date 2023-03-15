/**
 * \file ssadetector.hpp
 * \brief Plugin for parsing vpn_automaton traffic.
 * \author Jan Jirák jirakja7@fit.cvut.cz
 * \author Karel Hynek hynekkar@cesnet.cz
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
 * ALTERNATIVELY, provided that this notice is retained in full, this
 * product may be distributed under the terms of the GNU General Public
 * License (GPL) version 2 or later, in which case the provisions
 * of the GPL apply INSTEAD OF those given above.
 *
 * This software is provided as is'', and any express or implied
 * warranties, including, but not limited to, the implied warranties of
 * merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall the company or contributors be liable for any
 * direct, indirect, incidental, special, exemplary, or consequential
 * damages (including, but not limited to, procurement of substitute
 * goods or services; loss of use, data, or profits; or business
 * interruption) however caused and on any theory of liability, whether
 * in contract, strict liability, or tort (including negligence or
 * otherwise) arising in any way out of the use of this software, even
 * if advised of the possibility of such damage.
 *
 */

#ifndef IPXP_SSADETECTOR_HPP
#define IPXP_SSADETECTOR_HPP

#include <cstring>
#include <sstream>

#ifdef WITH_NEMEA
  #include "fields.h"
#endif

#include <ipfixprobe/process.hpp>
#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/packet.hpp>
#include <ipfixprobe/ipfix-elements.hpp>

namespace ipxp {

#define SSADETECTOR_UNIREC_TEMPLATE "SSA_CONF_LEVEL"

UR_FIELDS (
   uint8 SSA_CONF_LEVEL
)

#define SYN_RECORDS_NUM 100
#define PKT_TABLE_SIZE 91
#define MIN_PKT_SIZE 60
#define MAX_PKT_SIZE 150
#define MAX_TIME_WINDOW 3000000 // in microseconds
using dir_t = uint8_t;

/**
 * \brief Flow record extension header for storing parsed SSADETECTOR data.
 */
struct RecordExtSSADetector : public RecordExt {
   static int REGISTERED_ID;

   struct pkt_entry 
   {
      pkt_entry();
      void reset();
      timeval& get_time(dir_t dir);

      timeval ts_dir1;
      timeval ts_dir2;
      
   };

   struct pkt_table
   {
      public:
      pkt_entry table_[PKT_TABLE_SIZE];

      void reset();

      bool check_range_for_presence(uint16_t len, uint8_t down_by, dir_t dir, const timeval& ts_to_compare);
      void update_entry(uint16_t len, dir_t dir, timeval ts);

      private:
      static inline int8_t get_idx_from_len(uint16_t len);
      static inline bool time_in_window(const timeval& ts_now, const timeval& ts_old);
      inline bool entry_is_present(int8_t idx, dir_t dir, const timeval& ts_to_compare);
   };


   uint8_t possible_vpn {0}; // fidelity of this flow beint vpn
   uint64_t suspects {0};
   uint8_t syn_pkts_idx {0};
   uint8_t syn_pkts[SYN_RECORDS_NUM];

   pkt_table syn_table{};
   pkt_table syn_ack_table{};

   RecordExtSSADetector() : RecordExt(REGISTERED_ID)
   {
   }

   void reset ()
   {
      syn_table.reset();
      syn_ack_table.reset();
   }

#ifdef WITH_NEMEA
   virtual void fill_unirec(ur_template_t *tmplt, void *record)
   {
      ur_set(tmplt, record, F_SSA_CONF_LEVEL, possible_vpn);
   }

   const char *get_unirec_tmplt() const
   {
      return SSADETECTOR_UNIREC_TEMPLATE;
   }
#endif

   virtual int fill_ipfix(uint8_t *buffer, int size)
   {
      if (size < 1) {
         return -1;
      }
      buffer[0] = (uint8_t) possible_vpn;
      return 1;
   }

   const char **get_ipfix_tmplt() const
   {
      static const char *ipfix_template[] = {
         IPFIX_SSADETECTOR_TEMPLATE(IPFIX_FIELD_NAMES)
         NULL
      };
      return ipfix_template;
   }

   std::string get_text() const 
   {
      std::ostringstream out; 
      out << "SSA=" << (int)possible_vpn;
      return out.str();
   }
};

/**
 * \brief Process plugin for parsing SSADETECTOR packets.
 */
class SSADetectorPlugin : public ProcessPlugin
{
public:
   SSADetectorPlugin();
   ~SSADetectorPlugin();
   void init(const char *params);
   void close();
   OptionsParser *get_parser() const { 
      return new OptionsParser("SSADetector", "Check traffic for SYN-SYNACK-ACK sequence to find possible network tunnels."); 
      }
   std::string get_name() const { return "SSADetector"; }
   RecordExt *get_ext() const { return new RecordExtSSADetector(); }
   ProcessPlugin *copy();

   int pre_create(Packet &pkt);
   int post_create(Flow &rec, const Packet &pkt);
   int pre_update(Flow &rec, Packet &pkt);
   int post_update(Flow &rec, const Packet &pkt);
   void pre_export(Flow &rec);
   static inline void transition_from_init(RecordExtSSADetector *record, uint16_t len, 
                                           const timeval& ts, uint8_t dir);
   static inline void transition_from_syn(RecordExtSSADetector *record, uint16_t len, 
                                          const timeval& ts, uint8_t dir);
   static inline bool transition_from_syn_ack(RecordExtSSADetector *record, uint16_t len, 
                                              const timeval& ts, uint8_t dir);

};

}
#endif /* IPXP_SSADETECTOR_HPP */

