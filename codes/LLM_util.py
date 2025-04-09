import torch


def get_sys_prompt():
    sys_info = """You are now a discriminator of predicted human intentions in disaster scenarios, tasked with determining wether the given user's possible next intention is right based on user's previous intention sequence, disaster level, a possible next intention and other reference intention sequences.
Note that:
1.The intentions are token embeddings, it contains the intention information. It is wrapped in the "``", which means that any token in two "`" refers to the intention embedding rather than the text token.
2.Disaster level is divided into: 
(1)"no disaster": Indicates that there is no effect on human mobility.
(2)"minor disaster": An individual's daily travel plans may not be affected much, but alternatives should be considered.
(3)"general disaster": Human mobility may be affected by disasters, causing certain activities to be adjusted or cancelled for safety reasons.
(4)"severe disaster": Human movement will be greatly affected by the disaster, and the probability of staying still will increase.
3.Reference intention sequences are selected sequences from a intention sequence RAG. They are the most similar sequences to the given intention sequence. You can refer to these sequences to generate your answers, but don't copy them exactly.
4.The given possible intention is not necessarily accurate, you need to judge whether it it needed to change.
5.Both the user's previous intention sequence and the other reference intention sequences are in the format of a list like: [`intention embedding 1`, `intention embedding 2`...].
 """
    return sys_info


def get_instruct_prompt():
    ins_info = """Let's think step by step. You need to answer each of the three questions below, and if the answer to the first question is "yes", the following questions will be output as "None":
(1)Does the given possible next intent embedding right?
(2)If the answer to the previous question is "yes", this answer is set to "None". If the answer to the previous question is "no", please answer: Given the current disaster level, should the next intention be "stay still"?
(3)If the answer to the previous question is "yes", this answer is set to "stay still". If the answer to the previous question is "no", you need to give the index of the correct next intention embedding.
The indexes and embeddings of the intentions you can choose from are:
"""
    return ins_info


def get_tokenized_prompt(
    tokenizor,
    intention_emb_seq,
    ref_intention_emb_seq,
    disaster_level,
    similar_trajs,
    ret_next_intention_emb,
    intention_embs,
):
    # 0: text token, 1: intention token in intention modal, 2: intention token in text modal
    token_type = []
    prompt_seq = [
        get_sys_prompt(),
        f"Disaster Level: {disaster_level}. Intention embedding sequence: [",
    ]
    tokenizor_mask = [True, True]
    token_type = [0, 0]
    for index in range(intention_emb_seq.shape[0]):
        prompt_seq.append("`")
        tokenizor_mask.append(True)
        token_type.append(0)
        prompt_seq.append(intention_emb_seq[index].tolist())
        tokenizor_mask.append(False)
        token_type.append(1)
        prompt_seq.append("`,")
        tokenizor_mask.append(True)
        token_type.append(0)
    prompt_seq.append("]")
    tokenizor_mask.append(True)
    token_type.append(0)
    prompt_seq.append(
        "The given possible next intention embedding for this sequence is:`"
    )
    tokenizor_mask.append(True)
    token_type.append(0)
    prompt_seq.append(ret_next_intention_emb.tolist())
    tokenizor_mask.append(False)
    token_type.append(2)
    prompt_seq.append("`")
    tokenizor_mask.append(True)
    token_type.append(0)
    prompt_seq.append(
        "You need to refer to the following sequence to distinguish whether the given possible next intention embedding is right:"
    )
    tokenizor_mask.append(True)
    token_type.append(0)
    for index in range(len(ref_intention_emb_seq)):
        prompt_seq.extend(
            [
                f"Reference Sequence {index}:\n",
                f"Disaster level: {similar_trajs[index]['disaster_level']}\n",
                "Intention Sequence:[",
            ]
        )
        tokenizor_mask.extend([True, True, True])
        token_type.extend([0, 0, 0])
        for i in range(ref_intention_emb_seq[index].shape[0]):
            prompt_seq.append("`")
            tokenizor_mask.append(True)
            token_type.append(0)
            prompt_seq.append(ref_intention_emb_seq[index][i].tolist())
            tokenizor_mask.append(False)
            token_type.append(1)
            prompt_seq.append("`,")
            tokenizor_mask.append(True)
            token_type.append(0)
        prompt_seq.append("]\n")
        tokenizor_mask.append(True)
        token_type.append(0)
    prompt_seq.append(get_instruct_prompt())
    tokenizor_mask.append(True)
    token_type.append(0)
    for index, intention in enumerate(intention_embs):
        prompt_seq.append(str(index) + ":`")
        tokenizor_mask.append(True)
        token_type.append(0)
        prompt_seq.append(intention)
        tokenizor_mask.append(False)
        token_type.append(2)
        prompt_seq.append("`,\n")
        tokenizor_mask.append(True)
        token_type.append(0)
    prompt_seq.append(
        """Now give your answer. You should output the answer as a ["yes","yes","yes"] format, nothing else. Answer:"""
    )
    tokenizor_mask.append(True)
    token_type.append(0)
    tokens = []
    # sample_index = []
    # for index, x in enumerate(prompt_seq):
    #     if not tokenizor_mask[index]:
    #         sample_index.append("intenionxxx")
    #     else:
    #         sample_index.append(x)
    # sample = "".join(sample_index)
    for index in range(len(prompt_seq)):
        if tokenizor_mask[index]:
            token = tokenizor.encode_plus(
                prompt_seq[index],
                add_special_tokens=True,
                return_token_type_ids=False,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            if index == 0:
                tokens.append(token["input_ids"])
            else:
                tokens.append(token["input_ids"][:, 1:])
        else:
            token = torch.tensor(prompt_seq[index], dtype=torch.float32)
            tokens.append(token)
    return tokens, tokenizor_mask, token_type
