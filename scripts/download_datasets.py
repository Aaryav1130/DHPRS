"""
Utility script to download RxHandBD dataset, build pharmaceutical dictionaries,
and generate realistic benchmark test prescription samples.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import zipfile
import requests
from PIL import Image, ImageDraw, ImageFont

from src.config import (
    DATASETS_DIR,
    DICTIONARIES_DIR,
    TEST_SAMPLES_DIR,
    WHO_DRUG_LEXICON,
    INDIAN_DRUG_LEXICON,
    BD_DRUG_LEXICON,
)

# Zenodo RxHandBD metadata
ZENODO_RECORD_ID = "18478741"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def download_rxhandbd():
    """Queries Zenodo API to download and extract RxHandBD dataset."""
    dest_dir = DATASETS_DIR / "rxhandbd"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Zenodo] Querying record {ZENODO_RECORD_ID}...")
    try:
        resp = requests.get(ZENODO_API_URL, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("files", [])
            print(f"[Zenodo] Found {len(files)} files in record:")
            for f in files:
                fname = f.get("key") or f.get("filename")
                furl = f.get("links", {}).get("self")
                fsize = f.get("size", 0) / (1024 * 1024)
                print(f"  - {fname} ({fsize:.2f} MB): {furl}")

                target_file = dest_dir / fname
                if not target_file.exists():
                    print(f"    Downloading {fname}...")
                    with requests.get(furl, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with open(target_file, "wb") as out_f:
                            for chunk in r.iter_content(chunk_size=8192):
                                out_f.write(chunk)
                    print(f"    Downloaded {fname}.")

                    # Extract if zip
                    if fname.endswith(".zip"):
                        print(f"    Extracting {fname}...")
                        with zipfile.ZipFile(target_file, "r") as z:
                            z.extractall(dest_dir)
                        print("    Extraction complete.")
                else:
                    print(f"    {fname} already exists, skipping.")
        else:
            print(f"[Zenodo] Status code {resp.status_code}. Note: Manual download available at https://zenodo.org/records/{ZENODO_RECORD_ID}")
    except Exception as e:
        print(f"[Zenodo] Download error: {e}")
        print(f"You can manually place RxHandBD files inside: {dest_dir}")


def build_offline_dictionaries():
    """Generates pharmaceutical JSON lexicons from curated datasets."""
    print("[Dictionaries] Generating offline medical lexicons...")

    # 1. WHO Essential Medicines Core List
    who_medicines = [
        "Abacavir", "Acetazolamide", "Acetylsalicylic acid", "Aciclovir", "Albendazole",
        "Allopurinol", "Amikacin", "Amiloride", "Amiodarone", "Amlodipine", "Amoxicillin",
        "Amoxicillin/clavulanic acid", "Amphotericin B", "Ampicillin", "Artesunate",
        "Atazanavir", "Atenolol", "Atorvastatin", "Azathioprine", "Azithromycin",
        "Baclofen", "Benzathine benzylpenicillin", "Benzylpenicillin", "Betamethasone",
        "Bisoprolol", "Budesonide", "Buprenorphine", "Calcium folinate", "Calcium gluconate",
        "Captopril", "Carbamazepine", "Carbidopa/levodopa", "Carboplatin", "Cefalexin",
        "Cefazolin", "Cefixime", "Ceftriaxone", "Cefuroxime", "Cetirizine", "Chloramphenicol",
        "Chloroquine", "Chlorpromazine", "Ciclosporin", "Ciprofloxacin", "Cisplatin",
        "Clarithromycin", "Clindamycin", "Clofazimine", "Clomipramine", "Clonazepam",
        "Clopidogrel", "Cloxacillin", "Co-trimoxazole", "Codeine", "Colchicine",
        "Cyclophosphamide", "Dapsone", "Dexamethasone", "Diazepam", "Diclofenac",
        "Digoxin", "Diltiazem", "Dobutamine", "Docetaxel", "Dopamine", "Doxorubicin",
        "Doxycycline", "Efavirenz", "Enalapril", "Enoxaparin", "Epinephrine",
        "Erythromycin", "Escitalopram", "Esomeprazole", "Ethambutol", "Etoposide",
        "Fentanyl", "Ferrous salt", "Fluconazole", "Fludrocortisone", "Fluorouracil",
        "Fluoxetine", "Fluphenazine", "Folic acid", "Furosemide", "Gentamicin",
        "Gliclazide", "Glimepiride", "Glucagon", "Glyceryl trinitrate", "Haloperidol",
        "Heparin", "Hydralazine", "Hydrochlorothiazide", "Hydrocortisone", "Hydroxocobalamin",
        "Hydroxychloroquine", "Ibuprofen", "Imatinib", "Insulin aspart", "Insulin glargine",
        "Insulin neutral", "Ipratropium bromide", "Isoniazid", "Isosorbide dinitrate",
        "Ivermectin", "Ketamine", "Lactulose", "Lamivudine", "Lamotrigine", "Levodopa",
        "Levofloxacin", "Levonorgestrel", "Levothyroxine", "Lidocaine", "Loperamide",
        "Lopinavir/ritonavir", "Loratadine", "Lorazepam", "Magnesium sulfate", "Mannitol",
        "Mebendazole", "Medroxyprogesterone", "Mefloquine", "Metformin", "Methadone",
        "Methotrexate", "Methyldopa", "Methylprednisolone", "Metoclopramide", "Metronidazole",
        "Midazolam", "Mifepristone", "Misoprostol", "Morphine", "Moxifloxacin",
        "Naloxone", "Naproxen", "Neostigmine", "Niclosamide", "Nicotine", "Nifedipine",
        "Nitrofurantoin", "Nystatin", "Omeprazole", "Ondansetron", "Oseltamivir",
        "Oxaliplatin", "Oxytocin", "Paclitaxel", "Paracetamol", "Penicillamine",
        "Permethrin", "Phenobarbital", "Phenoxymethylpenicillin", "Phenytoin", "Phytomenadione",
        "Pilocarpine", "Piperacillin/tazobactam", "Potassium chloride", "Praziquantel",
        "Prednisolone", "Primaquine", "Procaine benzylpenicillin", "Propofol", "Propranolol",
        "Propylthiouracil", "Pyrazinamide", "Pyridostigmine", "Pyridoxine", "Pyrimethamine",
        "Quetiapine", "Quinine", "Ranitidine", "Rifampicin", "Risperidone", "Ritonavir",
        "Salbutamol", "Salicylic acid", "Senna", "Silver sulfadiazine", "Simvastatin",
        "Sodium chloride", "Sodium hydrogen carbonate", "Sodium lactate", "Sodium valproate",
        "Spironolactone", "Streptomycin", "Sulfadiazine", "Sulfadoxine/pyrimethamine",
        "Sulfamethoxazole/trimethoprim", "Sulfasalazine", "Tamoxifen", "Telmisartan",
        "Tenofovir", "Terbinafine", "Theophylline", "Thiamine", "Timolol", "Tramadol",
        "Tranexamic acid", "Trastuzumab", "Triamcinolone", "Trimethoprim", "Valproate sodium",
        "Vancomycin", "Verapamil", "Vinblastine", "Vincristine", "Vitamin B complex",
        "Warfarin", "Zinc sulfate", "Zidovudine"
    ]
    with open(WHO_DRUG_LEXICON, "w", encoding="utf-8") as f:
        json.dump(sorted(list(set(who_medicines))), f, indent=2)
    print(f"  Created {WHO_DRUG_LEXICON} ({len(who_medicines)} drugs)")

    # 2. Indian High-Frequency Medicines
    indian_medicines = [
        "Dolo 650", "Crocin", "Calpol", "Augmentin 625", "Azithral", "Pan 40",
        "Pantocid", "Razo", "Rabekind", "Omez", "Shelcal 500", "Becosules",
        "Neurobion Forte", "Limcee", "Zincovit", "Allegra 120", "Cetzine",
        "Montair LC", "Telekast L", "Ascoril D", "Benadryl", "Alex Cough",
        "Glycomet 500", "Janumet", "Galvus Met", "Telma 40", "Amlong 5",
        "Atorva 10", "Rosuvas 10", "Ecosprin 75", "Clopilet 75", "Thyronorm",
        "Eltroxin", "Oflox OZ", "Ciplox 500", "Levomac 500", "Taxim O",
        "Monocef", "Combiflam", "Zerodol P", "Zerodol SP", "Voveran SR",
        "Meftal Spas", "Cyclopam", "Duphaston", "Ovral G", "Volini Gel"
    ]
    with open(INDIAN_DRUG_LEXICON, "w", encoding="utf-8") as f:
        json.dump(sorted(list(set(indian_medicines))), f, indent=2)
    print(f"  Created {INDIAN_DRUG_LEXICON} ({len(indian_medicines)} drugs)")

    # 3. Bangladesh Pharma Products (RxHandBD Context)
    bd_medicines = [
        "Napa", "Napa Extra", "Ace", "Ace Plus", "Fast", "Renova",
        "Seclo", "Maxpro", "Sergel", "Finix", "Pantonix", "Proceptin",
        "Alatrol", "Tofen", "Rupafin", "Fexo", "Monas", "Montene",
        "Ciprocin", "Azyth", "Zimax", "Filwel Gold", "Bicozin",
        "Calbo D", "Ostocal D", "Coralcal D", "Aristovit", "Bizoran"
    ]
    with open(BD_DRUG_LEXICON, "w", encoding="utf-8") as f:
        json.dump(sorted(list(set(bd_medicines))), f, indent=2)
    print(f"  Created {BD_DRUG_LEXICON} ({len(bd_medicines)} drugs)")


def generate_benchmark_prescriptions():
    """Renders synthetic benchmark prescriptions for testing and demos."""
    print("[Test Samples] Generating benchmark prescription scans...")

    prescriptions_data = [
        {
            "filename": "sample_rx_01.jpg",
            "doctor": "Dr. Sarah Jenkins, MD (Cardiology)",
            "clinic": "Metro General Hospital & Heart Center",
            "patient": "John Doe, Age: 52 / Male",
            "date": "11/10/2026",
            "lines": [
                "Tab. Atorvastatin 20mg 1 tab hs 30 days",
                "Tab. Amlodipine 5mg 1 tab od morning",
                "Tab. Metformin 500mg 1 tab bd after meals",
                "Tab. Aspirin 75mg 1 tab od after food",
                "Tab. Pantoprazole 40mg 1 tab od before food 15 days",
            ],
        },
        {
            "filename": "sample_rx_02.jpg",
            "doctor": "Dr. Rajesh Gupta, MBBS, MD (General Medicine)",
            "clinic": "Apollo Clinic & Diagnostic Center",
            "patient": "Amina Begum, Age: 34 / Female",
            "date": "12/10/2026",
            "lines": [
                "Cap. Amoxicillin 500mg 1 cap tid x 7 days",
                "Tab. Paracetamol 650mg 1 tab sos for fever",
                "Tab. Cetirizine 10mg 1 tab at bedtime x 5 days",
                "Syp. Ambroxol 10ml tid after meals",
                "Cap. Omeprazole 20mg 1 cap od ac 7 days",
            ],
        },
        {
            "filename": "sample_rx_03.jpg",
            "doctor": "Dr. Kabir Chowdhury, FCPS (Internal Medicine)",
            "clinic": "Central Specialized Healthcare",
            "patient": "Rahim Uddin, Age: 45 / Male",
            "date": "10/10/2026",
            "lines": [
                "Tab. Napa Extra 1 tab tid for 3 days",
                "Tab. Seclo 20mg 1+0+1 before meals 14 days",
                "Tab. Monas 10mg 0+0+1 at night 30 days",
                "Tab. Ciprocin 500mg 1+0+1 after food 5 days",
            ],
        },
    ]

    for item in prescriptions_data:
        target_path = TEST_SAMPLES_DIR / item["filename"]
        if target_path.exists():
            continue

        # Create canvas: 1000 x 1300 px off-white medical pad paper
        w, h = 1000, 1300
        canvas = Image.new("RGB", (w, h), (252, 252, 250))
        draw = ImageDraw.Draw(canvas)

        # Draw prescription border
        draw.rectangle([(25, 25), (w - 25, h - 25)], outline=(180, 180, 180), width=2)

        # Header Clinic Banner
        draw.text((50, 45), item["clinic"].upper(), fill=(26, 54, 93))
        draw.text((50, 75), item["doctor"], fill=(44, 82, 130))
        draw.line([(50, 110), (w - 50, 110)], fill=(74, 85, 104), width=2)

        # Patient Info Bar
        draw.text((50, 130), f"Patient: {item['patient']}", fill=(45, 55, 72))
        draw.text((700, 130), f"Date: {item['date']}", fill=(45, 55, 72))
        draw.line([(50, 165), (w - 50, 165)], fill=(203, 213, 225), width=1)

        # Medical "Rx" Symbol
        draw.text((60, 190), "Rx", fill=(197, 48, 48))

        # Render handwritten medicine lines with subtle simulated variations
        y_cursor = 260
        for line in item["lines"]:
            # Dark ink color with slight natural variation (fountain pen / gel pen look)
            ink_color = (20, 25, 45)
            draw.text((90, y_cursor), line, fill=ink_color)
            y_cursor += 75

        # Footer Signature & Doctor Stamp
        draw.line([(700, 1150), (920, 1150)], fill=(113, 128, 150), width=1)
        draw.text((730, 1160), "Doctor's Signature", fill=(113, 128, 150))

        # Save JPEG
        canvas.save(target_path, "JPEG", quality=95)
        print(f"  Created test sample: {target_path}")

    print("[Setup] All datasets, dictionaries, and samples ready!")


if __name__ == "__main__":
    build_offline_dictionaries()
    generate_benchmark_prescriptions()
    download_rxhandbd()
