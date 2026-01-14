import pandas as pd

import json

PASS = "pass"


def generate_context(row):
    context_dict = {
        "features_path": row.get("UNI_patch_features_path", ""),
        # "coords_path": str(row["UNI_patch_features_path"]).replace('.pt', '.h5').replace('pt_files', 'h5_files'),
        "location_1": row.get("location_1", "").lower(),
        "location_2": row.get("location_2", "").lower(),
        "h_pylori": row.get("h_pylori", "").lower(),
        "severity": row.get("category", "").lower(),
        "gastritis_detail_present": row.get("gastritis_detail_present", "").lower(),
        "gastritis_type_add": row.get("gastritis_type_add", "").lower(),
        "gastritis_detail_grade": row.get("gastritis_detail_grade", "").lower(),
        "benign_type": row.get("benign_type", "").lower(),
        "dysplasia_type": row.get("dysplasia_type", "").lower(),
        "cancer_type": row.get("cancer_type", "").lower(),
        "cancer_detail": row.get("cancer_detail", "").lower(),
        "gastritis_type_sydney": row.get("gastritis_type_sydney", "").lower(),
        "set": row.get("Set", "").lower(),
        "final_caption": row.get("final_caption", "").lower(),
    }
    return context_dict


def generate_question_answer(context_dict, step):
    map_inx = ["location", "h_pylori", "severity", "gastritis_type", "gastritis_detail", "benign_type",
               "dysplasia_type", "cancer_type", "cancer_detail"]
    if context_dict["gastritis_type_add"] != "" and context_dict["gastritis_type_sydney"] != "":
        gastritis_type_sydney = [context_dict["gastritis_type_add"]] + context_dict["gastritis_type_sydney"].split(", ")
    elif context_dict["gastritis_type_sydney"] != "":
        gastritis_type_sydney = context_dict["gastritis_type_sydney"].split(", ")
    else:
        gastritis_type_sydney = [context_dict["gastritis_type_add"], "nothing"]
    gastic_grade = context_dict["gastritis_detail_grade"].split('.')[:-1]
    if len(gastic_grade) > 1:
        gastic_grade = [grade.strip() for grade in gastic_grade]
    L_list = [
        [context_dict["location_2"]],
        [context_dict["h_pylori"]],
        [context_dict["severity"]],
        gastritis_type_sydney,
        gastic_grade,
        [context_dict["benign_type"]],
        [context_dict["dysplasia_type"]],
        [context_dict["cancer_type"]],
        [context_dict["cancer_detail"]],
    ]

    cancer_detail_instruction = ""
    if len(context_dict["gastritis_detail_present"].split(",")) == 2:
        gastris_detail_instruction = f"The following features are present in this slide: {context_dict['gastritis_detail_present']}. Specify the inflammation grade for each as mild, moderate, or marked. Only evaluate chronic inflammation and neutrophilic activity if they are included in the list above." \
                                     "\nPlease format your answer as:'Chronic inflammation is {mild/moderate/marked}. Neutrophilic activity is {mild/moderate/marked}.'."
    else:
        gastris_detail_instruction = f"The following features are present in this slide: {context_dict['gastritis_detail_present']}. Specify the inflammation grade for each as mild, moderate, or marked. Only evaluate chronic inflammation and neutrophilic activity if they are included in the list above." \
                                     f"\nPlease format your answer as:'{context_dict['gastritis_detail_present'].capitalize()} " \
                                     "is {mild/moderate/marked}'."
    cancer_detail_task = ""
    if context_dict["cancer_type"] == "malignant lymphoma":
        cancer_detail_instruction = "'For malignant lymphoma, the tumor is {NOS/extranodal marginal zone lymphoma of malt}'."
        cancer_detail_task = ["NOS", "extranodal marginal zone lymphoma of malt"]
    elif context_dict["cancer_type"] == "carcinoma, adenocarcinoma":
        cancer_detail_instruction = "'For adenocarcinoma, the tumor is {well differentiated/moderately differentiated/poorly cohesive carcinoma}."
        cancer_detail_task = ["well differentiated", "moderately differentiated", "poorly cohesive carcinoma"]
    chronic_list = []
    neutrophilic_list = []
    if "chronic inflammation" in context_dict["gastritis_detail_present"]:
        chronic_list.append("chronic inflammation is mild")
        chronic_list.append("chronic inflammation is moderate")
        chronic_list.append("chronic inflammation is marked")
    if "neutrophilic activity" in context_dict["gastritis_detail_present"]:
        neutrophilic_list.append("neutrophilic activity is mild")
        neutrophilic_list.append("neutrophilic activity is moderate")
        neutrophilic_list.append("neutrophilic activity is marked")

    T_list = [

        ['antrum', 'body', 'cardia', 'fundus', 'prepylorus']
        ,
        ["negative", "positive"],
        [
            "inflammatory disease", "benign tumor", "dysplasia", "cancer"
        ],
        [
            "chronic inflammation", "neutrophilic activity", "glandular atrophy", "intestinal metaplasia",
            "erosion",
            "ulceration"
        ],
        [
            ["chronic inflammation is mild", "chronic inflammation is moderate", "chronic inflammation is marked"],
            ["neutrophilic activity is mild", "neutrophilic activity is moderate", "neutrophilic activity is marked"]
        ],
        [
            "hyperplastic polyp", "fundic gland polyp", "inflammatory polyp", "granulation tissue type polyp",
            "xanthoma", "gastritis cystica polyposa"
        ],
        [
            "low grade", "high grade",
            "indefinite"
        ],
        [
            "malignant lymphoma", "carcinoma, adenocarcinoma", "carcinoma, squamous cell carcinoma",
            "carcinoma, neuroendocrine tumor", "nos"
        ],
        cancer_detail_task
    ]
    Q_dict = {
        "location": "Identify both the general location (proximal vs. distal stomach) and the specific anatomical subregion of this slide?"
                    "\nThe proximal stomach includes the cardia, fundus, and body, while the distal  stomach includes the antrum and prepylorus."
                    "\nPlease format your answer as: "
                    "'This slide represents the {general location}, specifically the {subregion}.'.",
        "h_pylori": "Is this slide positive or negative for Helicobacter pylori infection?"
                    "\nPlease format your answer as: "
                    "'This slide is {positive/negative} for Helicobacter pylori infection.'.",
        "severity": f"Considering the spectrum of gastric pathology, determine the category of diagnosis represented in this slide: inflammatory disease, benign tumor, dysplasia, or cancer."
                    "\nPlease format your answer as: "
                    "'The category of diagnosis represented in this slide is {inflammatory disease/benign tumor/dysplasia/cancer}.'.",
        "gastritis_type": f"Evaluate this slide diagnosed as inflammatory disease for the following histologic features and indicate which are present:\nSydney system: chronic inflammation, neutrophilic activity, glandular atrophy, intestinal metaplasia."
                          f"\nAdditional features: erosion, ulceration."
                          "\nPlease format your answer as:"
                          "\n* If any Sydney system features are present: 'Among the Sydney system features, this slide shows [chronic inflammation, neutrophilic activity, glandular atrophy, intestinal metaplasia]. Additionally, this slide shows {erosion/ulceration}.'."
                          "\n* If no Sydney system features are present: 'This slide shows nothing. Additionally, this slide shows [erosion,ulceration]'."
                          "\n* If no additional features (neither erosion nor ulceration) are present, please omit the “Additionally” sentence.",
        "gastritis_detail": gastris_detail_instruction,
        "benign_type": f"This slide is diagnosed with a benign tumor gastric lesion. Identify the specific histologic type observed."
                       "\n Please format your answer as: "
                       "'This slide shows {fundic gland polyp"
                       "/hyperplastic polyp"
                       "/inflammatory polyp/"
                       "granulation tissue type polyp/"
                       "xanthoma/"
                       "gastritis cystica polyposa}.'.",
        "dysplasia_type": f"This slide shows dysplastic changes. Determine the grade of dysplasia , indefinite, low grade, or high grade."
                          "\nPlease format your answer as:"
                          "\n* For low or high grade dysplasia: 'The dysplasia is {low grade/high grade}, {tubular adenoma/tubulovillous adenoma}.'"
                          "\n* For indefinite dysplasia: 'The dysplasia is indefinite.'"
        ,
        "cancer_type": f"This slide shows malignant features. Determine the type of malignant tumor: carcinoma, malignant lymphoma, or NOS. If it is a carcinoma, also identify the histologic subtype."
                       "\nPlease format your answer as:"
                       "\n* For malignant lymphoma or NOS: 'The malignant tumor is a {malignant lymphoma/NOS}.'"
                       "\n* For carcinoma: 'The malignant tumor is a carcinoma, {adenocarcinoma/"
                       "squamous cell carcinoma/"
                       "neuroendocrine tumor}.'."
        ,
        "cancer_detail": f"Given the malignant tumor type {context_dict['cancer_type']}, specify the relevant detail or additional condition."
                         f"\nPlease format your answer as: {cancer_detail_instruction}"
    }
    if context_dict["h_pylori"] == "positive":
        Q_dict["severity"] = Q_dict["severity"].replace("this slide", "this Helicobacter pylori-positive slide")
        Q_dict["gastritis_type"] = Q_dict["gastritis_type"].replace("this slide",
                                                                    "this Helicobacter pylori-positive slide")
        Q_dict["gastritis_detail"] = Q_dict["gastritis_detail"].replace("this slide",
                                                                        "this Helicobacter pylori-positive slide")
        Q_dict["benign_type"] = Q_dict["benign_type"].replace("This slide", "This Helicobacter pylori-positive slide")
        Q_dict["dysplasia_type"] = Q_dict["dysplasia_type"].replace("This slide",
                                                                    "This Helicobacter pylori-positive slide")
        Q_dict["cancer_type"] = Q_dict["cancer_type"].replace("This slide", "This Helicobacter pylori-positive slide")
    A_dict = {
        "location": f"This slide represents the {context_dict['location_1']}, specifically the {context_dict['location_2']}.",
        "h_pylori": f"This slide is {context_dict['h_pylori']} for Helicobacter pylori infection.",
        "severity": f"The category of diagnosis represented in this slide is {context_dict['severity']}.",
        "gastritis_type": f"Among the Sydney system features, this slide shows " + (
            f"{context_dict['gastritis_type_sydney']}." if context_dict['gastritis_type_sydney'] else "nothing."
        ) + (
                              f" Additionally, this slide shows {context_dict['gastritis_type_add']}." if context_dict[
                                  'gastritis_type_add'] else ""
                          ),
        "gastritis_detail": f"{context_dict['gastritis_detail_grade']}",
        "benign_type": f"This slide shows {context_dict['benign_type']}.",
        "dysplasia_type": f"The dysplasia is {context_dict['dysplasia_type']}.",
        "cancer_type": f"The malignant tumor is a {context_dict['cancer_type']}.",
        "cancer_detail": f"For {context_dict['cancer_type']}, the tumor is {context_dict['cancer_detail']}."
    }

    return {"question": Q_dict[map_inx[step]],
            "answer": A_dict[map_inx[step]],
            "label": L_list[step],
            "task": T_list[step]
            }


def get_conversation(data):
    conversations = {
        "features_path": data["features_path"],
        "caption": data["final_caption"],
        "conversations": []
    }
    if data["location_1"] != PASS and data["location_2"] != PASS and data["location_1"] != "" and data[
        "location_2"] != "":
        conversations["conversations"].append(generate_question_answer(data, 0))

    if data["h_pylori"] != PASS and data["h_pylori"] != "":
        conversations["conversations"].append(generate_question_answer(data, 1))
    if data["severity"] != PASS and data["severity"] != "":
        conversations["conversations"].append(generate_question_answer(data, 2))
    severity = data["severity"]
    if severity == "inflammatory disease":
        conversations["conversations"].append(generate_question_answer(data, 3))
        if PASS not in data["gastritis_detail_grade"]:
            if data["gastritis_detail_present"] != "":
                gastritis_detail_present = data["gastritis_detail_present"].split(", ")
                grades = data["gastritis_detail_grade"].split("|")
                temp = ""
                for i, grade in enumerate(grades):
                    temp += f"{gastritis_detail_present[i]} is {grade.strip()}. "
                data["gastritis_detail_grade"] = temp.strip()
                conversations["conversations"].append(generate_question_answer(data, 4))
    elif severity == "benign tumor":
        if data["benign_type"] != PASS:
            conversations["conversations"].append(generate_question_answer(data, 5))
    elif severity == "dysplasia":
        if data["dysplasia_type"] != PASS:
            conversations["conversations"].append(generate_question_answer(data, 6))
    elif severity == "cancer":
        if data["cancer_type"] != PASS:
            conversations["conversations"].append(generate_question_answer(data, 7))
        if data["cancer_detail"] != PASS:
            if data["cancer_detail"] != "":
                conversations["conversations"].append(generate_question_answer(data, 8))
    return conversations


def generate_context_list(csv_path, data_set="Train"):
    df = pd.read_csv(csv_path)
    context_list = []
    for _, row in df.iterrows():
        row = row.fillna("")  # Fill NaN values with empty strings
        if data_set == "Test":
            if row["Set"] == data_set or row["Set"] == "Test_no_detail":

                context = generate_context(row)
                if context["h_pylori"] == "positive":
                    context_list.append(context)
        if row["Set"] == data_set == "Train":
            context = generate_context(row)
            context_list.append(context)

    vqa_set = {}
    i = 0
    for data in context_list:
        convs = get_conversation(data)
        vqa_set[f"{data['set']}::{i + 1}"] = convs
        i += 1
    return vqa_set


def generate_full_description_formatted(context_dict):
    # Compose each part in the answer format style:
    location_ans = ""
    if context_dict["location_1"] != PASS and context_dict["location_2"] != PASS:
        location_ans = f"Whole slide image represents the {context_dict['location_1']}, specifically the {context_dict['location_2']}."
    h_pylori_ans = ""
    if context_dict["h_pylori"] != PASS:
        h_pylori_ans = f"This slide is {context_dict['h_pylori']} for Helicobacter pylori infection."
    severity_ans = f"For diagnosis, the category of diagnosis represented in this slide is {context_dict['severity']}."
    gastritis_type_ans = ""
    gastritis_detail_ans = ""
    if context_dict["severity"] == "inflammatory disease":
        if context_dict["gastritis_type_sydney"] != "":
            gastritis_type_ans = f"Among the Sydney system features, this slide shows {context_dict['gastritis_type_sydney']}."
        else:
            gastritis_type_ans = "This slide shows nothing."
        if context_dict["gastritis_type_add"] != "":
            gastritis_type_ans += f" Additionally, this slide shows {context_dict['gastritis_type_add']}."
        if context_dict.get("gastritis_detail_grade", "") != "" and PASS not in context_dict.get(
                "gastritis_detail_grade", ""):
            gastritis_detail_ans = "The details of the inflammation are as follows: "
            gastritis_detail = context_dict.get("gastritis_detail_grade", "").split("|")
            gastric_detail_present = context_dict.get("gastritis_detail_present", "").split(", ")
            for i in range(len(gastritis_detail)):
                gastritis_detail_ans += f"{gastric_detail_present[i]} is {gastritis_detail[i].strip()},"
            gastritis_detail_ans = gastritis_detail_ans.rstrip(",") + "."
    benign_ans = ""
    if context_dict.get("benign_type", "") != PASS and context_dict.get("benign_type", "") != "":
        benign_ans = f"This slide shows {context_dict['benign_type']}."

    dysplasia_ans = ""
    if context_dict.get("dysplasia_type", "") != PASS and context_dict.get("dysplasia_type", "") != "":
        dysplasia_ans = f"The dysplasia is {context_dict['dysplasia_type']}."

    cancer_ans = ""
    if context_dict.get("cancer_type", "") != PASS and context_dict.get("cancer_type", "") != "":
        cancer_ans = f"The malignant tumor is a {context_dict['cancer_type']}."

    cancer_detail_ans = ""
    if context_dict.get("cancer_detail", "") != PASS and context_dict.get("cancer_detail", "") != "":
        cancer_detail_ans = f"For {context_dict['cancer_type']}, the tumor is {context_dict['cancer_detail']}."

    # Combine all parts, filtering out empty strings
    full_description = " ".join(p for p in [
        location_ans,
        h_pylori_ans,
        severity_ans,
        gastritis_type_ans,
        gastritis_detail_ans,
        benign_ans,
        dysplasia_ans,
        cancer_ans,
        cancer_detail_ans
    ] if p)

    return full_description



# data_set = "Train" # or "Test"

# csv_file = "path/CMCUJB_VQA_250827.csv"

# vqa = generate_context_list(csv_file, data_set)
# with open(f"path/CMCUJB_VQA_{data_set.lower()}.jsonl", "w") as f:
#     json.dump(vqa, f, indent=4)

