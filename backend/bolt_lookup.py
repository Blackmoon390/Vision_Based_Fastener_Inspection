import pandas as pd

import os


class BoltLookup:

    def __init__(self, excel_file):
        self.df = pd.read_excel(
            excel_file,
            sheet_name="OpenCV Lookup"
        )

    def find_bolt(self,
                  thread_diameter,
                  head_height,
                  across_corners):

        best = None
        best_score = 1e9

        for _, row in self.df.iterrows():

            d_score = abs(
                thread_diameter - row["Thread Diameter"]
            )

            h_score = abs(
                head_height - row["Head Height"]
            )

            ac_score = abs(
                across_corners - row["Across Corners"]
            )

            score = (
                d_score * 5 +
                h_score * 2 +
                ac_score * 1
            )

            if score < best_score:
                best_score = score
                best = row

        confidence = max(
            0,
            100 - best_score * 10
        )

        return {
            "bolt_size": best["Bolt Size"],
            "confidence": round(confidence, 2),
            "thread_diameter_db": best["Thread Diameter"],
            "head_height_db": best["Head Height"],
            "ac_db": best["Across Corners"]
        }
    


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DB_FILE = os.path.join(
#     BASE_DIR,
#     "ISO_Metric_Database",
#     "ISO_Metric_Hex_Bolt_Database.xlsx"
# )



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_FILE = os.path.join(
    BASE_DIR,
    "ISO_Metric_Database",
    "ISO_Metric_Hex_Bolt_Database.xlsx"
)
lookup = BoltLookup(DB_FILE)