import argparse
import random
import yaml
import os
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Sample a sequence of frames with shared visible vehicle")
    parser.add_argument("--datapath", type=str, required=True, help="Data package name, e.g. 2021_08_18_09_02_56")
    parser.add_argument("--step", type=int, required=True, help="Step size between frames")
    parser.add_argument("--frame_num", type=int, required=True, help="Number of frames to sample")
    return parser.parse_args()

def load_visibility_data(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def extract_frame_id(frame_name):
    return int(frame_name.replace(".yaml", ""))

def sample_sequence(data, step, frame_num):
    step = step if step % 2 == 0 else step + 1  # Ensure step is even

    # 按 ego_id 分组
    frame_ids = defaultdict(list)
    visible_other_ids = defaultdict(list)
    for entry in data:
        fid = extract_frame_id(entry['frame_name'])
        frame_ids[entry['ego_id']].append(fid)
        visible_other_ids[entry['ego_id']].append(entry["visible_other_ids"])

    step = round(step/2) * 2

    sample_pool = defaultdict(list)
    for ego_id in frame_ids:
        num_set = set(frame_ids[ego_id])

        for idx, frame_id in enumerate(frame_ids[ego_id]):
            if idx + step//2 * frame_num > len(frame_ids[ego_id]):
                continue
            seq = [ frame_id + i * step for i in range(frame_num)]

            if all([s in num_set for s in seq]):
                common_seen = set(visible_other_ids[ego_id][idx])
                for jdx in [idx + t * step//2 for t in range(1, frame_num)]:
                    # print(jdx, frame_ids[ego_id][jdx], common_seen)
                    common_seen &= set(visible_other_ids[ego_id][jdx])
                    if not common_seen:
                        break
                if common_seen:
                    sample_pool[ego_id].append(frame_id)
            
    # print(sample_pool)
    pairs = [(k,v) for k, values in sample_pool.items() for v in values]
    sampled = random.choice(pairs)

    ego_id, start_frame = sampled
    idx = frame_ids[ego_id].index(start_frame)
    frames = [frame_ids[ego_id][idx + i * step//2] for i in range(frame_num)]
    common_seen = set(visible_other_ids[ego_id][idx])
    for jdx in [idx + t * step//2 for t in range(1, frame_num)]:
        common_seen &= set(visible_other_ids[ego_id][jdx])
    other_id = random.choice(list(common_seen)) if common_seen else None
    result = {
        'ego_id': ego_id,
        'other_id': other_id,
        'frames': frames
    }
    return result

def main():
    args = parse_args()
    data = load_visibility_data(os.path.join(args.datapath, "mutual_visibility.yaml"))
    result = sample_sequence(data, args.step, args.frame_num)

    if result:
        print("Sampled Sequence:")
        print("ego_id:", result['ego_id'])
        print("other_id:", result['other_id'])
        print("frames:", result['frames'])
    else:
        print("[ERROR] No valid sequence found.")

if __name__ == "__main__":
    main()



