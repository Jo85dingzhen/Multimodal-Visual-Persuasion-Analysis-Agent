import os
import re
import base64
import csv
import json
import time
from openai import OpenAI

# === 1. Configuration ===
# Ensure your environment variable OPENAI_API_KEY is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️ WARNING: OPENAI_API_KEY not found. Please set it before running.")

client = OpenAI(api_key=api_key)

# Using GPT-4o for best Multimodal (Vision) performance
MODEL_NAME = "gpt-4o"

# Input/Output Directories
IMAGE_DIR = "images_from_docx"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "persuasion_analysis_results.csv")
OUTPUT_HTML = os.path.join(RESULTS_DIR, "visual_report.html")

# === 2. Expanded Persona Definitions (12 Types) ===
PERSONAS = [
    {"id": "P1_Traditionalist", "desc": "Values heritage, duty, and authority. Skeptical of modern hype.", "bias": "Prefers Authority, Familiarity."},
    {"id": "P2_Trendsetter", "desc": "Values social status, FOMO, and being first.", "bias": "Prefers Social Proof, Scarcity, Novelty."},
    {"id": "P3_Safety_Anxious", "desc": "Risk-averse, prioritizes stability and safety.", "bias": "Dislikes Scarcity (stressful). Prefers Trustworthiness, Neutrality."},
    {"id": "P4_Elite_Luxury", "desc": "Values exclusivity and uniqueness. Dislikes crowds.", "bias": "Prefers Scarcity (if exclusive). Dislikes Social Proof (common)."},
    {"id": "P5_Skeptic", "desc": "Needs evidence and data. Rejects emotional manipulation.", "bias": "Prefers Trustworthiness (Data). Dislikes Emotional Appeal."},
    {"id": "P6_Pragmatist", "desc": "Values utility and clear product views.", "bias": "Prefers Neutral/Clear shots. Dislikes artistic fluff."},
    {"id": "P7_Community", "desc": "Values belonging and family connection.", "bias": "Prefers Social Proof, Emotional Appeal."},
    {"id": "P8_Individualist", "desc": "Values uniqueness and personal identity.", "bias": "Prefers Personal Identity. Dislikes Social Proof."},
    {"id": "P9_Futurist", "desc": "Loves innovation and sci-fi aesthetics.", "bias": "Prefers Novelty. Dislikes Familiarity."},
    {"id": "P10_Naturalist", "desc": "Values purity, nature, and organic cues.", "bias": "Prefers Trustworthiness (Nature). Dislikes Artificiality."},
    {"id": "P11_Busy_Pro", "desc": "Values efficiency and availability.", "bias": "Prefers Availability. Dislikes lines/wait."},
    {"id": "P12_Jester", "desc": "Values entertainment and humor.", "bias": "Prefers Humor. Bored by clinical images."}
]

# Mapping Strategy ID to Name
PAIR_STRATEGY = {
    1: "Authority", 2: "Social Proof", 3: "Scarcity", 4: "Emotional Appeal",
    5: "Personal Identity", 6: "Humor", 7: "Trustworthiness", 8: "Familiarity", 9: "Novelty",
}

def encode_image(image_path: str) -> str:
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_pair(idx, strategy, img_a_path, img_b_path, persona):
    """
    Calls GPT-4o to analyze the image pair under a specific persona.
    """
    img_a_b64 = encode_image(img_a_path)
    img_b_b64 = encode_image(img_b_path)

    # === System Prompt: Persona Injection ===
    system_prompt = (
        f"You are a participant in a psychology experiment on visual persuasion.\n"
        f"You are NOT an AI assistant. You are a specific persona:\n"
        f"ID: {persona['id']}\n"
        f"Description: {persona['desc']}\n"
        f"Biases: {persona['bias']}\n\n"
        f"Adopt this persona completely. Your choices must reflect YOUR specific biases, "
        f"even if they contradict standard marketing logic. "
        f"If a strategy (like Scarcity) triggers your anxiety, REJECT IT."
    )

    # === User Prompt: The Tasks ===
    user_prompt = (
        f"Experiment Context: Testing strategy '{strategy}'.\n"
        f"You see Image A and Image B.\n\n"
        f"--- TASK 1: Decision ---\n"
        f"Which image is MORE persuasive to YOU?\n"
        f"--- TASK 2: Rationale ---\n"
        f"Explain WHY. What specific visual details (colors, people, objects) made you choose this? "
        f"Why did the other image fail for you?\n"
        f"--- TASK 3: Difficulty Simulation ---\n"
        f"Imagine a hypothetical 'Image C' (a Weak/Ambiguous version of this strategy).\n"
        f"Rank the difficulty of comparing: [A vs B], [A vs C], [B vs C] from Easiest to Hardest.\n\n"
        f"Output VALID JSON:\n"
        f"{{\n"
        f"  \"chosen_image\": \"A\" or \"B\",\n"
        f"  \"rationale\": \"...\",\n"
        f"  \"difficulty_ranking\": [\"Easiest\", \"Medium\", \"Hardest\"],\n"
        f"  \"difficulty_reason\": \"...\"\n"
        f"}}"
    )

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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Error for {persona['id']}: {e}")
        return None

def generate_html_report(results, pairs):
    """
    Generates a visual HTML report to see images alongside rationales.
    This avoids the 'Black Box' problem.
    """
    html = """
    <html>
    <head>
        <title>Visual Persuasion Analysis Report</title>
        <style>
            body { font-family: sans-serif; padding: 20px; background: #f4f4f4; }
            .pair-container { background: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .images { display: flex; gap: 20px; margin-bottom: 20px; }
            .img-box { flex: 1; text-align: center; }
            .img-box img { max-width: 100%; max-height: 300px; border: 1px solid #ddd; }
            .results-table { width: 100%; border-collapse: collapse; }
            .results-table th, .results-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .results-table th { background-color: #eee; }
            .choice-A { color: green; font-weight: bold; }
            .choice-B { color: blue; font-weight: bold; }
        </style>
    </head>
    <body>
    <h1>Persuasion Analysis: Visual Report</h1>
    """

    for pair_id in sorted(pairs.keys()):
        strategy = PAIR_STRATEGY.get(pair_id, "Unknown")
        img_a_rel = os.path.relpath(pairs[pair_id]["A"], RESULTS_DIR)
        img_b_rel = os.path.relpath(pairs[pair_id]["B"], RESULTS_DIR)
        
        # Note: For HTML images to show, we need to copy images to results or use absolute paths.
        # Here we assume images are in ../images_from_docx relative to the html file
        img_a_src = f"../{pairs[pair_id]['A']}"
        img_b_src = f"../{pairs[pair_id]['B']}"

        html += f"""
        <div class="pair-container">
            <h2>Pair {pair_id}: {strategy}</h2>
            <div class="images">
                <div class="img-box">
                    <h3>Image A (Treatment)</h3>
                    <img src="{img_a_src}" alt="Image A">
                </div>
                <div class="img-box">
                    <h3>Image B (Control)</h3>
                    <img src="{img_b_src}" alt="Image B">
                </div>
            </div>
            <table class="results-table">
                <tr>
                    <th>Persona</th>
                    <th>Choice</th>
                    <th>Rationale (The "Why")</th>
                    <th>Difficulty Rank (Mental Sim)</th>
                </tr>
        """
        
        # Filter results for this pair
        pair_results = [r for r in results if r["Pair_ID"] == pair_id]
        
        for res in pair_results:
            choice_class = "choice-A" if res["Choice"] == "A" else "choice-B"
            html += f"""
            <tr>
                <td><strong>{res['Persona_ID']}</strong></td>
                <td class="{choice_class}">{res['Choice']}</td>
                <td>{res['Rationale']}</td>
                <td>{res['Difficulty_Ranking']}</td>
            </tr>
            """
        
        html += "</table></div>"
    
    html += "</body></html>"
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✨ Visual Report generated: {OUTPUT_HTML}")

def main():
    print("=== Visual Persuasion Experiment (Visual + Analytical) ===")
    
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
            
    print(f"Found {len(pairs)} image pairs.")
    all_results = []

    # 2. Run Evaluation Loop
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pair_ID", "Strategy", "Persona_ID", "Choice", "Rationale", "Difficulty_Ranking", "Difficulty_Reasoning"])

        for idx in sorted(pairs.keys()):
            if "A" not in pairs[idx] or "B" not in pairs[idx]: continue
            
            strategy = PAIR_STRATEGY.get(idx, "Unknown")
            print(f"\n\n==========================================")
            print(f"🖼️  ANALYZING PAIR {idx}: {strategy}")
            print(f"==========================================")

            for persona in PERSONAS:
                # print(f"  > Persona: {persona['id']}...")
                
                data = analyze_pair(idx, strategy, pairs[idx]["A"], pairs[idx]["B"], persona)
                
                if data:
                    # === VISUALIZATION: Console Output ===
                    # "Opening the Black Box" in the terminal
                    choice_icon = "🟢" if data.get("chosen_image") == "A" else "🔵"
                    print(f"\n👤 {persona['id']}")
                    print(f"   {choice_icon} Choice: Image {data.get('chosen_image')}")
                    print(f"   💡 Rationale: {data.get('rationale')[:150]}...") # Show first 150 chars
                    
                    # Save to Memory
                    row = {
                        "Pair_ID": idx,
                        "Strategy": strategy,
                        "Persona_ID": persona['id'],
                        "Choice": data.get("chosen_image"),
                        "Rationale": data.get("rationale"),
                        "Difficulty_Ranking": str(data.get("difficulty_ranking")),
                        "Difficulty_Reasoning": data.get("difficulty_reason")
                    }
                    all_results.append(row)
                    
                    # Save to CSV
                    writer.writerow(row.values())

    # 3. Generate HTML Report
    generate_html_report(all_results, pairs)
    print("\n✅ Experiment Complete.")
    print(f"📄 CSV Data: {OUTPUT_CSV}")
    print(f"📊 HTML Visual Report: {OUTPUT_HTML} (Open this in your browser!)")

if __name__ == "__main__":
    main()