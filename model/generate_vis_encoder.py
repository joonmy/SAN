
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

configuration = MBartConfig.from_pretrained('./config.json')
mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.save_pretrained('./768-3/')



