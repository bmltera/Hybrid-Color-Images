import json
import re

def extract_object(prompt):
    """
    Extract the main one-word object from a prompt.
    The object is usually the key noun mentioned right after "of a/an".
    For complex prompts, simplify to a single easy-to-guess word.
    """
    text = prompt.lower().strip()
    
    # Find the part after "of a(n)" or "of"
    match = re.search(r'\bof\s+(?:a\s+|an\s+)?(.+)', text)
    if not match:
        words = text.split()
        return words[-1]
    
    remainder = match.group(1).strip()
    
    # Split on prepositions that come AFTER the object noun phrase
    # But NOT on participles like "perched" or "feeding" - those come after the noun
    # e.g. "macaw perched in a rainforest" -> we want "macaw", not to split before it
    
    # Strategy: get the first "noun phrase" from the remainder.
    # The noun phrase ends at the first preposition or participle.
    
    # Participles that indicate end of noun phrase (come after the subject)
    stop_words_after_noun = [
        'on', 'in', 'under', 'by', 'with', 'at', 'from', 'among', 'over',
        'near', 'through', 'underwater', 'above', 'behind', 'beside',
        'displaying', 'perched', 'reading', 'feeding', 'growing', 'sitting',
        'standing', 'flying', 'swimming', 'running', 'walking', 'lying',
        'hanging', 'floating', 'resting', 'lined', 'filled', 'covered',
        'made', 'full', 'releasing'
    ]
    
    words = remainder.split()
    
    # Find where the noun phrase ends
    noun_phrase_words = []
    for w in words:
        if w in stop_words_after_noun:
            break
        noun_phrase_words.append(w)
    
    if not noun_phrase_words:
        noun_phrase_words = [words[0]] if words else [remainder]
    
    subject = ' '.join(noun_phrase_words)
    
    # Known adjectives/descriptors to skip
    adjectives = {
        'grey', 'gray', 'old', 'wooden', 'cracked', 'shattered', 'broken',
        'plain', 'bare', 'silhouetted', 'limestone', 'sandstone',
        'woolen', 'crumpled', 'fishing', 'grocery', 'snow-covered',
        'bengal', 'scarlet', 'mandarin', 'black', 'orange', 'red', 'blue',
        'green', 'violet-backed', 'hand-knitted', 'norwegian', 'brightly',
        'painted', 'exotic', 'tropical', 'california', 'double', 'monarch',
        'barn', 'reef', 'festival', 'midsummer', 'dense', 'tall',
        'lush', 'terraced', 'fruiting', 'sand', 'coral', 'sea',
        'pile', 'bundle', 'bouquet', 'bowl', 'rainbow', 'field', 'road',
        'asphalt', 'bright', 'metallic', 'neon', 'colored', 'simple',
        'highly', 'detailed', 'color', 'vibrant', 'digital'
    }
    
    subj_words = subject.split()
    nouns = [w for w in subj_words if w.lower().replace('-', '') not in adjectives]
    
    if nouns:
        obj = nouns[-1]
    else:
        obj = subj_words[-1] if subj_words else subject
    
    # Clean up hyphens
    if '-' in obj:
        parts = obj.split('-')
        obj = parts[-1]
    
    # Handle some edge cases
    # "person reading under a tree" -> person
    # "bowl of fruit in shadow" - remainder after first "of" is "a bowl of fruit..."
    # Actually the first match gives us "bowl of fruit in shadow"
    # noun phrase ends at "in" -> "bowl of fruit"  wait, "of" is not a stop word
    # Let me add "of" as a stop word... no, that would break everything since we already
    # split on "of" at the top level.
    # Actually the regex match finds the FIRST "of", so for 
    # "a pencil drawing of a bowl of fruit in shadow" it gives "bowl of fruit in shadow"
    # and noun phrase = ["bowl", "of", "fruit"] (stops at "in") -> last noun = "fruit"
    # That's actually fine! "fruit" is a reasonable object for that prompt.
    
    # Clean up plurals - simple de-pluralize
    no_strip = {'glass', 'grass', 'moss', 'dress', 'canvas', 'iris', 'ibis',
                'bus', 'lotus', 'cactus', 'compass', 'octopus', 'walrus',
                'asparagus'}
    if obj not in no_strip and len(obj) > 2 and obj.endswith('s') and not obj.endswith('ss'):
        obj = obj[:-1]
    
    # Post-processing fixups
    fixups = {
        'dune': 'dunes',
        'shear': 'shears',
        'stone': 'stones',
        'pebble': 'pebbles',
        'blossom': 'blossom',
        'berrie': 'berries',
        'butterflie': 'butterfly',
        'asparagu': 'asparagus',
        'cherrie': 'cherry',
        'bird': 'birds',
        'release': 'lantern',
        'bloom': 'garden',
        'teddie': 'teddy',
        'layer': 'cliff',
        'interior': 'cave',
        'clearing': 'jungle',
        'sanctuary': 'birds',
        'wildflower': 'flowers',
        'stick': 'sticks',
        'crack': 'pavement',
        'handle': 'door',
        'banana': 'banana',
        'truck': 'truck',
        'teddie': 'teddy',
    }
    if obj in fixups:
        obj = fixups[obj]
    
    return obj


def process():
    # Read all three files  
    # Check if backup exists - if so, read original 1.json from backup
    import os
    backup_path = os.path.join('backup', '1.json')
    if os.path.exists(backup_path):
        with open(backup_path, 'r', encoding='utf-8') as f:
            data1_original = json.load(f)
        print("Using backup/1.json for original data")
    else:
        with open('1.json', 'r', encoding='utf-8') as f:
            data1_full = json.load(f)
        # Extract only the original 35 items
        original_numbers = {810, 840, 841, 853, 874, 883, 885, 890, 911, 919, 928, 940, 960,
                            970, 997, 1000, 1006, 1008, 1010, 1022, 1034, 1040, 1042, 1057,
                            1058, 1066, 1076, 1079, 1100, 1101, 1102, 1104, 1114, 1214, 1230}
        data1_original = {"data": [item for item in data1_full['data'] if item['number'] in original_numbers]}
        print(f"Extracted {len(data1_original['data'])} original items from 1.json")
    
    with open('2.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    with open('3.json', 'r', encoding='utf-8') as f:
        data3 = json.load(f)
    
    valid_qualities = {'L', 'M', 'H'}
    
    # Filter 2.json and 3.json
    filtered_2 = [item for item in data2['data'] if item.get('quality', '') in valid_qualities]
    filtered_3 = [item for item in data3['data'] if item.get('quality', '') in valid_qualities]
    
    print(f"2.json: {len(data2['data'])} total -> {len(filtered_2)} with quality L/M/H")
    print(f"3.json: {len(data3['data'])} total -> {len(filtered_3)} with quality L/M/H")
    
    def process_item(item):
        grey_obj = extract_object(item['greyscale'])
        color_obj = extract_object(item['color'])
        return {
            "number": item['number'],
            "greyscale": item['greyscale'],
            "color": item['color'],
            "quality": item['quality'],
            "grey_object": grey_obj,
            "color_object": color_obj
        }
    
    existing_processed = [process_item(item) for item in data1_original['data']]
    new_items = [process_item(item) for item in filtered_2 + filtered_3]
    
    combined = existing_processed + new_items
    
    print(f"\n1.json original: {len(existing_processed)} items")
    print(f"New items from 2.json + 3.json: {len(new_items)}")
    print(f"Total combined: {len(combined)}")
    
    # Print ALL original items for verification
    print("\n--- All original 1.json entries ---")
    for item in existing_processed:
        print(f"  #{item['number']}: grey_object=\"{item['grey_object']}\", color_object=\"{item['color_object']}\"")
        print(f"    grey: {item['greyscale']}")
        print(f"    color: {item['color']}")
    
    # Print some new items
    print("\n--- First 15 new entries from 2/3.json ---")
    for item in new_items[:15]:
        print(f"  #{item['number']}: grey_object=\"{item['grey_object']}\", color_object=\"{item['color_object']}\"")
        print(f"    grey: {item['greyscale']}")
        print(f"    color: {item['color']}")
    
    # Write to 1.json
    output = {"data": combined}
    with open('1.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"\nDone! 1.json now has {len(combined)} items.")


if __name__ == '__main__':
    process()
