#!/bin/bash
# Resolve ${NETWORK_INTERFACE} in cyclone_config.xml and export CYCLONEDDS_URI
# pointing to the generated file. Source this script; do not execute it directly.
# CycloneDDS versions in RoboStack do not expand env-var placeholders in XML,
# so the substitution must happen before the node starts.
: "${NETWORK_INTERFACE:=enp0s31f6}"
_gen_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sed "s|\${NETWORK_INTERFACE}|${NETWORK_INTERFACE}|g" \
    "${_gen_script_dir}/cyclone_config.xml" \
    > /tmp/cyclone_config_resolved.xml
export CYCLONEDDS_URI="file:///tmp/cyclone_config_resolved.xml"
unset _gen_script_dir
