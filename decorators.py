import torch
import random
from typing import List, Tuple, Dict, Optional

class DataPipeline:
  # def __init__(self, annot_mp, label_foodtype_mp):
  def __init__(self):
    # self.annot_mp = annot_mp
    # self.label_foodtype_mp = label_foodtype_mp

    self.REVERSE_SUFFIX = " in reverse"
    self.SPEED_SUFFIX = " in {}x speed"
    self.MASK_SUFFIX = " the given video is a subpart of a full video"

  def __check_out(self, out):
    if out == None:
      raise Exception("[ERR] the output is None")

    if len(out) != 2:
      raise Exception("[ERR] the output is not of length 2")

    if type(out[0]) != torch.Tensor:
      raise Exception("[ERR] the first element of the output is not of type torch.Tensor")

    if type(out[1]) != str:
      raise Exception("[ERR] the second element of the output is not of type str")

    if out[1] == "":
      raise Exception("[ERR] the second element of the output is empty")

    if len(out[0]) == 0:
      raise Exception("[ERR] the first element of the output is empty")

  def _to_device(self,
                 t: torch.Tensor,
                 device: Optional[torch.device]):
    return t if device is None else t.to(device, non_blocking=True)

  def reverse_chunk(self,
                    frames: torch.Tensor,
                    caption: str,
                    *,
                    device: Optional[torch.device] = None) -> Tuple[torch.Tensor, str]:
    frames = self._to_device(frames, device)
    caption = caption + self.REVERSE_SUFFIX

    frames = frames.flip(0)
    out = (frames, caption)
    self.__check_out(out)

    return out

  def change_speed(self,
                   frames: torch.Tensor,
                   caption: str,
                   speed: float,
                   *,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    frames = self._to_device(frames, device)

    out = None

    if speed == 1:
      out = (frames, caption)

    elif speed > 1:
      speed = int(round(speed))
      caption = caption + self.SPEED_SUFFIX.format(str(speed))
      frames = frames[::speed]
      out = (frames, caption)

    else:
      repeat = max(1, int(round(1/speed)))
      caption = caption + self.SPEED_SUFFIX.format(str(speed))
      frames = frames.repeat_interleave(repeat, dim=0)
      out = (frames, caption)

    self.__check_out(out)
    return out

  def fps_sampling(self,
                   frame_chunks: List[torch.Tensor],
                   caption_chunks: List[str],
                   keep_ratio: float = 0.5,
                   *,
                   device: Optional[torch.device] = None) -> torch.Tensor:

    frame_chunks = [self._to_device(chunk, device) for chunk in frame_chunks]

    combined_frames = torch.cat(frame_chunks, dim=0)
    combined_captions = " ".join(caption_chunks)
    step = int(round(1 / keep_ratio))

    combined_frames = combined_frames[::step]
    out = (combined_frames, combined_captions)
    self.__check_out(out)

    return out

  def combine_chunks_reverse(self,
                             frame_chunks: List[torch.Tensor],
                             caption_chunks: List[str],
                             *,
                             device: Optional[torch.device] = None) -> torch.Tensor:
    frame_chunks = [self._to_device(chunk, device) for chunk in frame_chunks]
    combined_reverse = torch.cat(frame_chunks, dim=0).flip(0)
    caption_chunks = " ".join(caption_chunks) + self.REVERSE_SUFFIX
    out = (combined_reverse, caption_chunks)
    self.__check_out(out)
    return out

  def combine_mask_with_caption(self,
                                frame_chunks: List[torch.Tensor],
                                caption_chunks: List[str],
                                mask_ratio: float=0.3,
                                *,
                                device: Optional[torch.device] = None) -> torch.Tensor:

    assert len(frame_chunks) == len(caption_chunks), "Need caption per chunk"
    frame_chunks = [self._to_device(chunk, device) for chunk in frame_chunks]

    # build potentioal caption once
    caption = " ".join(caption_chunks) + self.MASK_SUFFIX

    video = torch.cat(frame_chunks, dim=0)

    # which whole chunks to mask?

    num_mask = int(round(len(frame_chunks) * mask_ratio))

    if num_mask:
      to_mask = random.sample(range(len(frame_chunks)), num_mask)
      lengths = torch.tensor([c.shape[0] for c in frame_chunks], device=video.device)
      offsets = torch.cat((torch.zeros(1, device=video.device, dtype=torch.long), lengths.cumsum(0)))

      keep = torch.ones(offsets[-1].item(), dtype=torch.bool, device=video.device)
      for idx in to_mask:                                # usually a handful; loop is fine
          keep[offsets[idx]: offsets[idx + 1]] = False   # mark frames to drop

      video = video[keep]
      # for idx in to_mask:
      #   video[offsets[idx]: offsets[idx + 1]] = 0

    out = (video, caption)
    self.__check_out(out)

    return out


# ------------------------------------------------------------------------------------------

class Compose:
  def __init__(self, transforms: List, *, device: Optional[torch.device] = None):
    self.transforms = transforms
    self.device = device

  def __call__(self, sample: Dict):
    if self.device is not None:
      if type(sample) == dict:
        sample["video"] = sample["video"].to(self.device, non_blocking=True)
      elif type(sample) == list:
        for i in range(len(sample)):
          sample[i]["video"] = sample[i]["video"].to(self.device, non_blocking=True)

    for t in self.transforms:
      sample = t(sample)

    return sample

DA = DataPipeline()

class RandomReverse:
  def __init__(self, p: float = 0.5):
    self.p = p

  def __call__(self, sample: Dict):
    if random.random() < self.p:
      sample["video"], sample["caption"] = DA.reverse_chunk(frames = sample["video"],
                                                            caption = sample["caption"])

    return sample


class RandomChangeSpeed:
  def __init__(self, speeds=(0.5, 1, 2), p: float=0.5):
    self.speeds = speeds
    self.p = p

  def __call__(self, sample: Dict):
    if random.random() < self.p:
      sp = random.choice(self.speeds)
      sample["video"], sample["caption"] = DA.change_speed(frames = sample["video"],
                                                           caption = sample["caption"],
                                                           speed = sp)

    return sample

class FPSSampling:
  def __init__(self, keep_ratio: float = 0.5):
    self.keep_ratio = keep_ratio

  def __call__(self, sample_ls: List[Dict]) -> Dict:
    if len(sample_ls) > 1:
      videos = []
      captions = []

      for ind in len(sample_ls):
        videos.append(sample_ls[ind]["video"])
        captions.append(sample_ls[ind]["caption"])

      out = DA.fps_sampling(frame_chunks = videos,
                          caption_chunks = captions,
                          keep_ratio = self.keep_ratio)

      return {"video": out[0], "caption": out[1]}



class CombineReverse:
  def __call__(self, sample_ls: List[Dict]):
    if len(sample_ls) > 1:
      videos = []
      captions = []

      for ind in range(len(sample_ls)):
        videos.append(sample_ls[ind]["video"])
        captions.append(sample_ls[ind]["caption"])

      out = DA.combine_chunks_reverse(frame_chunks = videos,
                                      caption_chunks = captions)

      return {"video": out[0], "caption": out[1]}


class CombineMask:
  def __init__(self, mask_ratio: float = 0.3):
    self.mask_ratio = mask_ratio

  def __call__(self, sample_ls: List[Dict]):
    if len(sample_ls) > 1:
      videos = []
      captions = []

      for ind in range(len(sample_ls)):
        videos.append(sample_ls[ind]["video"])
        captions.append(sample_ls[ind]["caption"])

      out = DA.combine_mask_with_caption(frame_chunks = videos,
                                         caption_chunks = captions,
                                         mask_ratio = self.mask_ratio)

      return {"video": out[0], "caption": out[1]}


