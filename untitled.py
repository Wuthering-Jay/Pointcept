from pointcept.datasets.defaults import DefaultDataset

if __name__=="__main__":
    dataset=DefaultDataset(
        split="train",
        data_root="G:\\data\\DALES\\dales_las", # 数据集根目录
    )
    
    for i,data in enumerate(dataset):
        print(i)
        print(data["coord"].shape)
        print(data["segment"].shape)
        print(data["instance"].shape)
        print(data["name"])
        print(data["split"])
        if i>10:
            break