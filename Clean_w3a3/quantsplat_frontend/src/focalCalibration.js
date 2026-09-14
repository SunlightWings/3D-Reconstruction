// Canonical asset identifier: W3A3 with corrected camera rotation and focal calibration.
// The supplied bundle contains nine held-out camera renders, not an orbital turntable.
const frames = {
  "apple/110_13051_23361": [2, 24, 52, 75, 102, 125, 153, 175, 202],
  "ball/123_14363_28981": [2, 24, 52, 75, 102, 125, 153, 175, 202],
  "bowl/70_5792_13401": [2, 24, 52, 75, 102, 125, 153, 175, 202],
  "broccoli/412_56288_108844": [2, 24, 52, 75, 102, 125, 153, 175, 202],
  "hydrant/167_18184_34441": [2, 24, 52, 74, 101, 125, 153, 175, 202],
  "remote/350_36761_68623": [2, 24, 52, 75, 102, 125, 153, 175, 202],
  "teddybear/187_20215_38541": [2, 24, 52, 75, 102, 125, 153, 175, 202],
  "toaster/372_41229_82130": [2, 24, 52, 75, 102, 125, 153, 175, 202],
};

export const focalCalibration = {
  id: "w3a3_qfix_focal",
  label: "W3A3 + focal calibration",
  description: "W3A3 renders with corrected camera rotation and focal calibration.",
  hasScene: (sceneId) => Boolean(frames[sceneId]),
  frameFor: (sceneId, index) => frames[sceneId]?.[index],
  pathFor: (sceneId, index) => {
    const frame = frames[sceneId]?.[index];
    return frame == null ? null : `images/${sceneId}/w3a3_qfix_focal/frame${String(frame).padStart(6, "0")}.png`;
  },
};
