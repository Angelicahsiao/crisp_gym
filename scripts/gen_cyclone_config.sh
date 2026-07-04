#!/bin/bash
# Resolve ${NETWORK_INTERFACE} in cyclone_config.xml and export CYCLONEDDS_URI
# pointing to the generated file. Source this script; do not execute it directly.
# CycloneDDS versions in RoboStack do not expand env-var placeholders in XML,
# so the substitution must happen before the node starts.
: "${NETWORK_INTERFACE:=enp0s31f6}"
# Use /sys/class/net rather than `ip`/`ifconfig`: the pixi env has no iproute2.
# The interface must exist AND be up: CycloneDDS cannot bind to an interface
# with no address (e.g. cable unplugged) and node creation fails.
if [ ! -e "/sys/class/net/$NETWORK_INTERFACE" ]; then
    echo "NETWORK_INTERFACE '$NETWORK_INTERFACE' not found, falling back to 'lo'."
    NETWORK_INTERFACE=lo
elif [ "$(cat /sys/class/net/$NETWORK_INTERFACE/operstate 2>/dev/null)" != "up" ] \
        && [ "$NETWORK_INTERFACE" != "lo" ]; then
    echo "NETWORK_INTERFACE '$NETWORK_INTERFACE' is not up (cable unplugged?), falling back to 'lo'."
    NETWORK_INTERFACE=lo
fi
_gen_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Substitute both placeholder spellings so the template works whether it uses
# ${NETWORK_INTERFACE} or ${ROS_NETWORK_INTERFACE}.
sed -e "s|\${NETWORK_INTERFACE}|${NETWORK_INTERFACE}|g" \
    -e "s|\${ROS_NETWORK_INTERFACE}|${NETWORK_INTERFACE}|g" \
    "${_gen_script_dir}/cyclone_config.xml" \
    > /tmp/cyclone_config_resolved.xml
export CYCLONEDDS_URI="file:///tmp/cyclone_config_resolved.xml"
unset _gen_script_dir
