import os
import re
import base64
import csv
import json
import time
import random
from openai import OpenAI, RateLimitError, APIError

# === 1. Configuration ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️ WARNING: OPENAI_API_KEY not found.")

client = OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4o"

# Input/Output Directories
IMAGE_DIR = "images_from_docx"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "final_analysis_results.csv")
OUTPUT_HTML = os.path.join(RESULTS_DIR, "final_visual_report.html")

# === 2. Personas ===
PERSONAS = [
    {"id": "P1_Traditionalist", "desc": "Values heritage, duty.", "bias": "Prefers Authority, Familiarity."},
    {"id": "P2_Trendsetter", "desc": "Values status, FOMO.", "bias": "Prefers Social Proof, Scarcity."},
    {"id": "P3_Safety_Anxious", "desc": "Risk-averse.", "bias": "Dislikes Scarcity. Prefers Trustworthiness."},
    {"id": "P4_Elite_Luxury", "desc": "Values exclusivity.", "bias": "Prefers Scarcity. Dislikes Social Proof."},
    {"id": "P5_Skeptic", "desc": "Needs evidence.", "bias": "Prefers Data. Dislikes Emotional Appeal."},
    {"id": "P6_Pragmatist", "desc": "Values utility.", "bias": "Prefers Clear shots. Dislikes artistic fluff."},
    {"id": "P7_Community", "desc": "Values belonging.", "bias": "Prefers Social Proof."},
    {"id": "P8_Individualist", "desc": "Values uniqueness.", "bias": "Dislikes Social Proof."},
    {"id": "P9_Futurist", "desc": "Loves innovation.", "bias": "Prefers Novelty."},
    {"id": "P10_Naturalist", "desc": "Values purity.", "bias": "Dislikes Artificiality."},
    {"id": "P11_Busy_Pro", "desc": "Values efficiency.", "bias": "Prefers Availability."},
    {"id": "P12_Jester", "desc": "Values humor.", "bias": "Prefers Humor."}
]

PAIR_STRATEGY = {
    1: "Authority", 2: "Social Proof", 3: "Scarcity", 4: "Emotional Appeal",
    5: "Personal Identity", 6: "Humor", 7: "Trustworthiness", 8: "Familiarity", 9: "Novelty",
}

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_pair_sequential(idx, strategy, img_a_path, img_b_path, persona):
    """
    Analyzes a single pair with strict Error Handling and Retries.
    """
    img_a_b64 = encode_image(img_a_path)
    img_b_b64 = encode_image(img_b_path)

    system_prompt = (
        f"You are a specific persona: {persona['id']} ({persona['desc']}). "
        f"Bias: {persona['bias']}. Adopt this persona completely."
    )

    user_prompt = (
        f"Context: Strategy '{strategy}'. Compare Image A and B.\n"
        f"1. Which is more persuasive to YOU?\n"
        f"2. Why? (Rationale)\n"
        f"3. Mental Sim: Rank difficulty of [A vs B], [A vs C (Weak)], [B vs C].\n"
        f"Output JSON: {{ \"chosen_image\": \"A\", \"rationale\": \"...\", \"difficulty_ranking\": [], \"difficulty_reason\": \"...\" }}"
    )

    # Retry Logic (Up to 5 times)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b_b64}"}},
                        ],
                    }
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content: return None
            
            data = json.loads(content)
            
            # Print specific choice to console for visualization
            choice = data.get("chosen_image")
            icon = "🟢" if choice == "A" else "🔵"
            print(f"   {icon} {persona['id']} chose Image {choice}")

            return {
                "Pair_ID": idx,
                "Strategy": strategy,
                "Persona_ID": persona['id'],
                "Choice": choice,
                "Rationale": data.get("rationale"),
                "Difficulty_Ranking": str(data.get("difficulty_ranking")),
                "Difficulty_Reasoning": data.get("difficulty_reason"),
                "Status": "Success"
            }

        except RateLimitError:
            wait_time = (5 * (attempt + 1)) + random.uniform(1, 3)
            print(f"   ⚠️ Rate Limit Hit. Cooling down for {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        except APIError as e:
            print(f"   ⚠️ API Error: {e}. Retrying...")
            time.sleep(5)
            
        except Exception as e:
            print(f"   ❌ Fatal Error: {e}")
            return None
            
    print(f"   ❌ Failed after {max_retries} attempts.")
    return None

def generate_html_report(results, pairs):
    html = """<html><head><style>
    body{font-family:sans-serif;padding:20px;background:#f4f4f4}
    .pair{background:white;padding:20px;margin-bottom:20px;border-radius:8px}
    table{width:100%;border-collapse:collapse;margin-top:10px}
    td,th{border:1px solid #ddd;padding:8px}
    th{background:#eee}
    .choice-A{color:green;font-weight:bold} .choice-B{color:blue;font-weight:bold}
    </style></head><body><h1>Final Analysis Report</h1>"""
    
    # Group results by Pair ID
    results.sort(key=lambda x: (x["Pair_ID"], x["Persona_ID"]))

    for pair_id in sorted(pairs.keys()):
        strategy = PAIR_STRATEGY.get(pair_id, "Unknown")
        pair_res = [r for r in results if r["Pair_ID"] == pair_id]
        
        # Relative path for HTML
        img_a = f"../{pairs[pair_id]['A']}"
        img_b = f"../{pairs[pair_id]['B']}"
        
        html += f"<div class='pair'><h2>Pair {pair_id}: {strategy}</h2>"
        html += f"<div style='display:flex;gap:20px'><div style='text-align:center'><b>Image A</b><br><img src='{img_a}' height='200'></div>"
        html += f"<div style='text-align:center'><b>Image B</b><br><img src='{img_b}' height='200'></div></div>"
        html += "<table><tr><th>Persona</th><th>Choice</th><th>Rationale</th></tr>"
        
        for res in pair_res:
            c_class = "choice-A" if res.get("Choice") == "A" else "choice-B"
            html += f"<tr><td>{res['Persona_ID']}</td><td class='{c_class}'>{res.get('Choice')}</td><td>{res.get('Rationale')}</td></tr>"
        html += "</table></div>"
    
    html += "</body></html>"
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✨ Visual Report generated: {OUTPUT_HTML}")

def main():
    print("=== 🛡️ Final Experiment: Sequential Processing (No Crashes) ===")
    
    # 1. Discover Pairs
    pattern = re.compile(r".*pair\s*(\d+).*([abAB])\.(png|jpg|jpeg)$", re.IGNORECASE)
    pairs = {}
    
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: {IMAGE_DIR} not found.")
        return

    for fname in os.listdir(IMAGE_DIR):
        m = pattern.match(fname)
        if m:
            idx, side = int(m.group(1)), m.group(2).upper()
            if idx not in pairs: pairs[idx] = {}
            pairs[idx][side] = os.path.join(IMAGE_DIR, fname)
            
    print(f"Found {len(pairs)} pairs. Starting analysis...")

    # 2. Run Analysis Loop (Sequential)
    all_results = []
    
    # Open CSV in append mode, or write header if new
    write_header = not os.path.exists(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Pair_ID", "Strategy", "Persona_ID", "Choice", "Rationale", "Difficulty_Ranking", "Difficulty_Reasoning", "Status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # Iterate Pairs
        for idx in sorted(pairs.keys()):
            if "A" not in pairs[idx] or "B" not in pairs[idx]: continue
            
            strategy = PAIR_STRATEGY.get(idx, "Unknown")
            print(f"\n--- Analyzing Pair {idx}: {strategy} ---")
            
            # Iterate Personas
            for persona in PERSONAS:
                result = analyze_pair_sequential(idx, strategy, pairs[idx]["A"], pairs[idx]["B"], persona)
                
                if result:
                    writer.writerow(result) # Save immediately to disk
                    f.flush()               # Force write
                    all_results.append(result)
                
                # IMPORTANT: Small sleep to be kind to the API limit
                time.sleep(1)

    # 3. Generate HTML
    generate_html_report(all_results, pairs)
    print("\n✅ Done! Analysis complete.")

if __name__ == "__main__":
    main()