from __future__ import annotations

from typing import Dict

DATASET_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "CIC-IDS2017": {
        "Src IP": "flow.src_ip",
        "Dst IP": "flow.dst_ip",
        "Src Port": "flow.src_port",
        "Dst Port": "flow.dst_port",
        "Protocol": "flow.protocol",
        "Label": "label.class",
    },
    "CIC-IDS2018": {
        "Src IP": "flow.src_ip",
        "Dst IP": "flow.dst_ip",
        "Src Port": "flow.src_port",
        "Dst Port": "flow.dst_port",
        "Protocol": "flow.protocol",
        "Label": "label.class",
    },
    "UNSW-NB15": {
        "srcip": "flow.src_ip",
        "dstip": "flow.dst_ip",
        "sport": "flow.src_port",
        "dsport": "flow.dst_port",
        "proto": "flow.protocol",
        "label": "label.class",
    },
    "CTU-13": {
        "src_ip": "flow.src_ip",
        "dst_ip": "flow.dst_ip",
        "src_port": "flow.src_port",
        "dst_port": "flow.dst_port",
        "proto": "flow.protocol",
        "label": "label.class",
    },
}

def get_dataset_field_map(dataset_name: str) -> Dict[str, str]:
    return DATASET_FIELD_MAPS.get(dataset_name, {})