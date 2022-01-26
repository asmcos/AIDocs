import yaml,os

folder = os.path.abspath(".")

def load_config_file(filename):
    f = open(filename)
    content = f.read()
    f.close()
    
    data = yaml.full_load(content)
    return data

#读取配置文件
book = load_config_file(folder + "/mkdocs.yml")
print(book['site_name'])

#获取书的目录列表，左侧菜单
nav = book['nav']

file_list = []
def parse_nav(content): 
    # 文件名
    if isinstance(content,str):
        print(content)
        file_list.append(content)

    # 子目录
    if isinstance(content,list):
        for i in content:
            parse_nav(list(i.values())[0])

# 解析书目录
for i in nav:
    parse_nav(list(i.values())[0])

print(file_list)

# 将文件内容存储到数据库里面
def file_to_db():
    orig_dir = 'docs/'
    for i in file_list:
        content = open(orig_dir+i).read()

#将数据库的内容存到目录文件
def db_to_file():
    obj_dir = ".tmp/docs/"
    for i in file_list:
        f = open(obj_dir+i,"w+")
        f.write("fds")
        f.close()

#file_to_db()

