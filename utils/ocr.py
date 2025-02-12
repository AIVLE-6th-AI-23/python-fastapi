import easyocr

readers = {
    'ko': easyocr.Reader(['ko', 'en']),     # 한국어
    'ru': easyocr.Reader(['ru', 'en']),     # 러시아어
    'vi': easyocr.Reader(['vi', 'en']),     # 베트남어
    'fr': easyocr.Reader(['fr', 'en']),     # 불어
    'ja': easyocr.Reader(['ja', 'en']),     # 일본어
    'zh': easyocr.Reader(['ch_sim', 'en']), # 중국어 간체
    'ar': easyocr.Reader(['ar', 'en']),     # 아랍어
    'hi': easyocr.Reader(['hi', 'en'])      # 힌디어
}

async def try_all_readers(image):
    best_result = {'text': '', 'confidence': 0, 'lang': ''}

    for lang, reader in readers.items():
        try:
            text_results = reader.readtext(image)
            if text_results:
                confidence = sum(result[2] for result in text_results) / len(text_results)
                extracted_text = ' '.join([result[1] for result in text_results])

                if confidence > best_result['confidence']:
                    best_result = {'text': extracted_text, 'confidence': confidence, 'lang': lang}
        except Exception :
            raise
        
    return best_result
