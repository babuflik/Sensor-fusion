#!/bin/bash

echo "🔍 Network Interface Discovery"
echo "============================"

# Function to get IP addresses
get_ips() {
    if command -v ip >/dev/null 2>&1; then
        ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1'
    elif command -v ifconfig >/dev/null 2>&1; then
        ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'
    else
        echo "❌ Could not find 'ip' or 'ifconfig' command."
        exit 1
    fi
}

IPS=$(get_ips)
COUNT=$(echo "$IPS" | wc -l)

if [ -z "$IPS" ]; then
    echo "❌ No network interfaces found (excluding localhost)."
    echo "   You can try running with --localhost if testing locally."
    exit 1
fi

echo "Found the following IP addresses:"
i=1
declare -A IP_MAP
DEFAULT_CHOICE=1
PRIORITY_FOUND=0

for ip in $IPS; do
    HINT=""
    if [[ $ip == 192.168.* ]]; then
        HINT=" (Likely LAN IP)"
        # High priority
        DEFAULT_CHOICE=$i
        PRIORITY_FOUND=2
    elif [[ $ip == 172.* ]]; then
        HINT=" (Likely WSL IP)"
        # Medium priority (only if we haven't found a high priority one)
        if [ $PRIORITY_FOUND -lt 2 ]; then
            DEFAULT_CHOICE=$i
            PRIORITY_FOUND=1
        fi
    fi
    echo "  [$i] $ip$HINT"
    IP_MAP[$i]=$ip
    ((i++))
done

# Add option for 0.0.0.0
echo "  [$i] 0.0.0.0 (Bind to all interfaces - Recommended if unsure)"
IP_MAP[$i]="0.0.0.0"
# If we haven't found a specific LAN/WSL IP, default to 0.0.0.0 (safest fallback)
if [ $PRIORITY_FOUND -eq 0 ]; then
    DEFAULT_CHOICE=$i
fi
((i++))

# Add option for manual entry
echo "  [$i] Manually enter IP"
MANUAL_OPT=$i

echo ""
echo "Which IP should the server bind to?"
echo "Tip: If connecting from a phone on the same Wi-Fi, choose the LAN IP (often 192.168.x.x)."
echo "Tip: If using WSL 2, choose the WSL IP (often 172.x.x.x) or 0.0.0.0."
echo ""

read -p "Enter number [1-$MANUAL_OPT] or press ENTER for default [$DEFAULT_CHOICE - ${IP_MAP[$DEFAULT_CHOICE]}]: " choice

if [ -z "$choice" ]; then
    choice=$DEFAULT_CHOICE
fi

if [ "$choice" -eq "$MANUAL_OPT" ]; then
    read -p "Enter IP address: " SELECTED_IP
else
    SELECTED_IP=${IP_MAP[$choice]}
fi

if [ -z "$SELECTED_IP" ]; then
    echo "❌ Invalid selection."
    exit 1
fi

echo ""
echo "✅ Selected IP: $SELECTED_IP"
echo "   Use this IP in your phone app settings."
echo ""

# Export for other scripts to use if sourced
export SENSOR_FUSION_IP=$SELECTED_IP
