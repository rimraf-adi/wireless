import yaml
import re
import os

def main():
    # Setup paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    tex_file = os.path.join(base_dir, 'radio_network_planning_kagal.tex')
    yaml_file = os.path.join(base_dir, 'theory.yaml')

    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            theory_blocks = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        return

    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {tex_file}: {e}")
        return

    for key, new_text in theory_blocks.items():
        if not new_text:
            continue
            
        new_text = new_text.strip()
        
        # Regex to target text securely enclosed within our markers
        pattern = re.compile(r'(%%% BEGIN: ' + re.escape(key) + r' %%%\n).*?(\n%%% END: ' + re.escape(key) + r' %%%)', re.DOTALL)
        
        if pattern.search(content):
            # Using lambda to avoid regex escaping issues with backslashes (common in LaTeX)
            content = pattern.sub(lambda m, text=new_text: m.group(1) + text + m.group(2), content)
        else:
            print(f"Warning: Could not find markers for '{key}' in {os.path.basename(tex_file)}")

    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Successfully updated {os.path.basename(tex_file)} with content from {os.path.basename(yaml_file)}")

if __name__ == '__main__':
    main()
