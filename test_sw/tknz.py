from transformers import AutoTokenizer
toknzr = AutoTokenizer.from_pretrained("./model/")


text="<|im_start|>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。<|im_end|>"
# b = text.encode("utf-8")
# garbled = b.decode("latin-1")

# # garbled="é"
# # # 先按 latin-1 编码成字节，再按 utf-8 解码
# # original = garbled.encode("latin-1").decode("utf-8")

# print(garbled)

# ids = toknzr.encode(text)
# print(ids)            # token id 序列
# print(toknzr.decode(ids))  


encoding = toknzr(
            text,  
            max_length=32,  # 限制最大长度
            padding="max_length",  # 不足部分补pad
            truncation=True,  # 超出部分截断
            return_tensors="pt",  # 返回PyTorch tensor形式（包含batch维度）
        )
print(encoding)

