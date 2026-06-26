"""
Pixel-to-mm calibration module
"""

class Calibration:
    def __init__(
        self,
        ref_height_mm,
        ref_height_px,
        ref_width_mm,
        ref_width_px
    ):
        self.height_scale = ref_height_mm / ref_height_px
        self.width_scale = ref_width_mm / ref_width_px

    def px_to_mm(self, measurements):
        return {
            "bolt_height_mm":
                round(measurements["bolt height"] * self.height_scale, 2),

            "bolt_thread_height_mm":
                round(measurements["bolt thread height"] * self.height_scale, 2),

            "bolt_total_width_mm":
                round(measurements["bolt total width"] * self.width_scale, 2),

            "bolt_center_width_mm":
                round(measurements["bolt center width (thread)"] * self.width_scale, 2),

            "bolt_bottom_width_mm":
                round(measurements["bolt bottom width (thread)"] * self.width_scale, 2)
        }