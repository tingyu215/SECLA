import json
import os
from matplotlib import pyplot as plt
from numpy import median


def get_score_by_name(test_allname_dir, name, max_num):
    # extract all file names end with 1000steps.json
    file_names = []
    for names in os.listdir(test_allname_dir):
        if names.endswith("1000steps.json"):
            file_names.append(names)

    if len(file_names) > max_num:
        file_names = sorted(file_names)[:max_num]
    
    score_dict = dict.fromkeys(range(len(file_names)), {})

    for i, dict_name in enumerate(sorted(file_names)):
        print(dict_name)
        with open(os.path.join(test_allname_dir, dict_name)) as f:
            allname_dict = json.load(f)
        
        score_list = []
        diff_list = []
        for sample in allname_dict.values():
            if [name] in sample["name_list"]:
                sim_scores = sample["sim_face_name"]
                name_idx = sample["name_list"].index([name])
                score_list.append(sim_scores[name_idx][name_idx])
                # matched name score - unmatched name score
                diff_list.append(sim_scores[name_idx][name_idx] - sim_scores[name_idx][1-name_idx])
        score_dict[i] = [score_list, diff_list]
    return score_dict


def make_plot_score_change(score_dict, name, out_dir, add_noname):
    x_data = [1000 * k for k in range(len(score_dict))]

    avg_score = []
    median_score = []
    avg_diff = []
    # avg_score_wrong = []

    for sample in score_dict.values():
        avg_score.append(sum(sample[0]) / len(sample[0]))
        avg_diff.append(sum(sample[1]) / len(sample[1]))
        median_score.append(median(sample[0]))
        # avg_score_wrong.append( sum([a-b for a,b in zip(sample[0],sample[1])]) / len([a-b for a,b in zip(sample[0],sample[1])]) )
    
    plt.plot(x_data, avg_score, label= "avg sim. score")
    plt.plot(x_data, avg_diff, label= "avg diff.")
    plt.plot(x_data, median_score, label= "median sim. score")
    # plt.plot(x_data, avg_score_wrong, label= "avg sim. score wrong name")
    plt.legend()
    plt.xlabel("#. of Steps")
    plt.ylabel("Similarity Scores")
    if add_noname:
        plt.title(f"Changes in similarity scores for {name} with NONAME added")
    else:
        plt.title(f"Changes in similarity scores for {name} without NONAME added")

    plt.savefig(os.path.join(out_dir, f"{name}_NONAME-{add_noname}.png"))
    
    plt.close()


if __name__ == "__main__":

    test_allname_dir = "/export/home2/NoCsBack/working/tingyu/face_naming/celeb/test_allname"
    test_allname_dir_noname = "/cw/working-frodo/tingyu/face_naming/celeb/test_allname"
    name = "Abigail Breslin"
    score_dict = get_score_by_name(test_allname_dir, name, max_num=5)
    score_dict1 = get_score_by_name(test_allname_dir_noname, name, max_num=5)
    
    make_plot_score_change(score_dict, name, out_dir = test_allname_dir, add_noname=False)
    make_plot_score_change(score_dict1, name, out_dir = test_allname_dir, add_noname=True)

