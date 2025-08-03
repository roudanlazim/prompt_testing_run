# Shipment Status Classification - Status-Only Evaluation

This script performs shipment classification by analyzing scan sequences (`ScanGroups`) and answering 18 predefined logistics questions per shipment using a single GPT-4o call.

Each model response is parsed and mapped into human-readable labels using an enum mapping file. The final answers are saved into a structured CSV output.

---

## What This Script Does

- Reads a CSV file containing a `ScanGroups` column with scan event lists
- Builds a structured prompt from a rule file and appends enforced formatting instructions
- Sends each scan group list to GPT-4o and receives a JSON response with answers to 18 questions
- Validates that the model response includes all required keys in the correct format
- Converts the raw model output into human-readable labels using a predefined mapping
- Writes the 18 answers into new columns in the output CSV, one column per question
- Tracks prompt and completion token usage and logs retry attempts per row

---

## Inputs

- A CSV file with a `ScanGroups` column (list of scan labels per shipment)
- A rules text file describing classification logic
- A JSON file defining enum mappings for all questions and answers
- A configuration file specifying:
  - Input and output file paths
  - Prompt rule path
  - Test selection parameters (row count or row indexes)

---

## Outputs

- A CSV file with 18 new columns (one per classification question)
- Each output value is a mapped, human-readable label (e.g., "Address issue", "No", "2")
- A `total_token_count` column reflecting GPT token usage per row
- A `.jsonl` log file tracking prompt and completion token counts and retry behavior

---

## Questions Answered

1. Why did the collection fail?
2. How many collection attempts were made?
3. What is the collection scheduling status?
4. What is the shipment customs status?
5. Why is the shipment held in customs?
6. Where was the shipment delivered?
7. Why did the delivery fail?
8. Was the delivery refused?
9. How many delivery attempts were made?
10. If the delivery was rescheduled, what is true?
11. Is the shipment discarded, abandoned, or lost?
12. Why is the delivery on hold?
13. Why is the shipment delayed in transit?
14. Why is the shipment on hold in transit?
15. Is the delivery rescheduled?
16. If the package was returned, what is true?
17. If the package is being returned, what is true?
18. What is the return outcome?

---

## How to Run

1. Configure `mono_config_settings.json` with:
   - Input CSV path
   - Prompt rule file
   - Enum mapping file
   - Row selection settings (optional)

2. Execute:

```bash
python run_mon_test_status_only.py
