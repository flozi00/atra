from aaas.model_utils import get_model_and_processor
from PIL import Image


def inference_ocr(data, mode, config):
    model, processor = get_model_and_processor(mode, "ocr", config)

    image = Image.open(data).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text
