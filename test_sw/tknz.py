from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./model/")


text="鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。"
b = text.encode("utf-8")
garbled = b.decode("latin-1")

# garbled="é"
# # 先按 latin-1 编码成字节，再按 utf-8 解码
# original = garbled.encode("latin-1").decode("utf-8")

print(garbled)

ids = tok.encode(text)
print(ids)            # token id 序列
print(tok.decode(ids))  