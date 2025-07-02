import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pointcept.datasets.las_dataloader import LasDataset

if __name__ == "__main__":
    
    dataset = LasDataset(
        split="test",
        data_root=r"E:\data\dales_las",
        transform=None,
        test_mode=False,
    )
    
    print(f"Dataset contains {len(dataset)} point clouds")
    
    # Get a sample point cloud
    print(len(dataset))

    for idx in range(len(dataset)):
        data_dict = dataset[idx]
        print(f"Data {idx}: {data_dict['name']}, Split: {data_dict['split']}, Points: {data_dict['coord'].shape[0]}")
        # Print information about the point cloud
        print(f"\nPoint cloud: {data_dict['name']}")
        print(f"Split: {data_dict['split']}")
        print(f"Points: {data_dict['coord'].shape[0]}")