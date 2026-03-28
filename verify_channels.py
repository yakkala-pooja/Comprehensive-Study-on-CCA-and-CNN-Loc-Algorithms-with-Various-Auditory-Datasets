import json

with open('channel_names.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("CHANNEL COUNT VERIFICATION")
print("=" * 60)
print(f"DAS: JSON says {data['das']['count']} channels, array has {len(data['das']['channels'])} items")
print(f"Fulsang EEG: JSON says {data['fulsang']['count']['eeg']} channels, array has {len(data['fulsang']['eeg_channels'])} items")
print(f"Fulsang EOG: JSON says {data['fulsang']['count']['eog']} channels, array has {len(data['fulsang']['eog_channels'])} items")
print(f"Fulsang Total: JSON says {data['fulsang']['count']['total']} total channels, all_channels array has {len(data['fulsang']['all_channels'])} items")
print()

print("=" * 60)
print("DAS DATASET - EXACT MAPPING")
print("=" * 60)
for i, ch in enumerate(data['das']['channels']):
    print(f"Channel {i:2d} -> {ch}")

print()
print("=" * 60)
print("FULSANG DATASET - EXACT MAPPING")
print("=" * 60)
for i, ch in enumerate(data['fulsang']['all_channels']):
    ch_type = "EEG" if i < len(data['fulsang']['eeg_channels']) else "EOG/Aux"
    print(f"Channel {i:2d} -> {ch:8s} ({ch_type})")








