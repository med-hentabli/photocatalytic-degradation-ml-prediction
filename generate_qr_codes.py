from pathlib import Path
import qrcode

STREAMLIT_URL = "https://photocatalytic-degradation-ml-prediction-fxexreoiber4vbbrlbhx4.streamlit.app/"
GITHUB_URL = "https://github.com/med-hentabli/photocatalytic-degradation-ml-prediction"

OUT = Path("qr_codes")
OUT.mkdir(exist_ok=True)

def save_qr(url: str, filename: str) -> None:
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=18,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(OUT / filename)

save_qr(STREAMLIT_URL, "streamlit_app_qr.png")
save_qr(GITHUB_URL, "github_repository_qr.png")

print(f"Saved QR codes to: {OUT.resolve()}")
