def get_seg(seg_group, morpho):
    allseg = []
    for currGroup in seg_group.includes:
        allseg.extend(
            get_seg(
                [x for x in morpho.segment_groups if x.id == currGroup.segment_groups][0],
                morpho,
            )
        )

    for currSeg in seg_group.members:
        allseg.append(currSeg.segments)

    return allseg
