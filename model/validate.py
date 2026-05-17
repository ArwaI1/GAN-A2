import calendar
import datetime
import argparse
from typing import List, Tuple

DAYS = ["[MON]", "[TUE]", "[WED]", "[THU]", "[FRI]", "[SAT]", "[SUN]"]
MONTHS = ["[JAN]", "[FEB]", "[MAR]", "[APR]", "[MAY]", "[JUN]", "[JUL]", "[AUG]", "[SEP]", "[OCT]", "[NOV]", "[DEC]"]

def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def validate_prediction(line: str) -> Tuple[int, int, str]:
    parts = line.strip().split()
    if len(parts) != 5: return 0, 4, "Invalid format"
    
    c_day, c_month, c_leap, c_decade = parts[0], parts[1], parts[2], parts[3]
    date_str = parts[4]
    
    try:
        d, m, y = map(int, date_str.split('-'))
        date_obj = datetime.datetime(y, m, d)
    except ValueError:
        return 0, 4, "Invalid calendar date (e.g., Feb 30)"

    score = 0
    errors = []

    # 1. Check Decade
    expected_decade = int(c_decade.strip('[]'))
    if str(y)[:3] == str(expected_decade): score += 1
    else: errors.append(f"Decade mismatch (Expected {expected_decade}0s, got {y})")

    # 2. Check Month
    if MONTHS[m-1] == c_month: score += 1
    else: errors.append(f"Month mismatch (Expected {c_month}, got {MONTHS[m-1]})")

    # 3. Check Leap Year
    expected_leap = True if c_leap == "[True]" else False
    if is_leap_year(y) == expected_leap: score += 1
    else: errors.append(f"Leap mismatch (Expected {expected_leap}, got {is_leap_year(y)})")

    # 4. Check Day of Week
    actual_day_idx = date_obj.weekday() # Monday is 0
    if DAYS[actual_day_idx] == c_day: score += 1
    else: errors.append(f"Day mismatch (Expected {c_day}, date is actually a {DAYS[actual_day_idx]})")

    return score, 4, ", ".join(errors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Prediction file to validate")
    args = parser.parse_args()

    total_score, total_conditions = 0, 0
    perfect_examples, failed_examples = [], []

    with open(args.file, 'r') as f:
        for line in f:
            score, possible, error_msg = validate_prediction(line)
            total_score += score
            total_conditions += possible
            
            if score == 4 and len(perfect_examples) < 3:
                perfect_examples.append(line.strip())
            elif score <= 3 and len(failed_examples) < 3 and "Invalid calendar" not in error_msg:
                failed_examples.append((line.strip(), error_msg))

    print(f"\n=== VALIDATION REPORT FOR {args.file} ===")
    print(f"Overall Accuracy: {(total_score/total_conditions)*100:.2f}%\n")
    
    print("🌟 PERFECT EXAMPLES (Model learned well):")
    for ex in perfect_examples: print(f"  + {ex}")
    
    print("\n⚠️ FAILED EXAMPLES (Model struggled):")
    for ex, reason in failed_examples: 
        print(f"  - {ex}\n    Reason: {reason}")
    print("\nReflection: Generative networks easily learn static categorical rules (Decades, Months), but struggle with cyclical algorithmic logic (Zeller's Congruence for Days of the week) without explicit supervision.")