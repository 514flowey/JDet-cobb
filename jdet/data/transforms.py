import random
import jittor as jt 
import cv2
import numpy as np
import math
from jdet.utils.registry import build_from_cfg,TRANSFORMS

@TRANSFORMS.register_module()
class Compose:
    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform,dict):
                transform = build_from_cfg(transform,TRANSFORMS)
            elif not callable(transform):
                raise TypeError('transform must be callable or a dict')
            self.transforms.append(transform)

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        
        return image, target

class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if target is not None and random.random() < self.prob:
            image = F.rotate( image, 90, expand=True )
            target = target.rotate90()
        return image, target

# (0, 90, 180, or 270)
class RandomRotateAug:
    def __init__(self, random_rotate_on):
        self.random_rotate_on = random_rotate_on

    def __call__( self, image, target=None ):
        if target is not None and self.random_rotate_on:
            indx = int(random.random() * 100) // 25
            image = F.rotate( image, 90 * indx, expand=True )
            for _ in range(indx):
                target = target.rotate90()
        return image, target

@TRANSFORMS.register_module()
class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        # NOTE Mingtao
        if w <= h:
          size = np.clip( size, int(w / 1.5), int(w * 1.5) )
        else:
          size = np.clip( size, int(h / 1.5), int(h * 1.5) )

        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def _resize_boxes(self,target,size):
        for key in ["bboxes","rboxes","polygons"]:
            if key not in target:
                continue
            bboxes = target[key]
            width,height = target["img_size"]
            new_w,new_h = size
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)
            target[key]=bboxes
    
    def _resize_mask(self,target,size):
        pass

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = image.resize(size[::-1],Image.BILINEAR)
        if target is not None:
            self._resize_boxes(target,image.size)
            target["img_size"]=image.size
        return image, target

# The image must be a square as we do not expand the image
class RandomSquareRotate( object ):
    def __init__( self, do=True ):
        self.do = do

    def __call__( self, image, target=None ):
        if target is None or not self.do:
            return image, target

        w, h = image.size
        assert w == h

        cx = w / 2
        cy = cx

        degree = random.uniform(0, 360)
        radian = degree * math.pi / 180

        new_image = image.rotate( -degree )

        sin = math.sin( radian )
        cos = math.cos( radian )

        masks = target.get_field( "masks" )
        polygons = list( map( lambda x: x.polygons[0], masks.instances.polygons ) )
        polygons = torch.stack( polygons, 0 ).reshape( (-1, 2) ).t()

        M = torch.Tensor([[cos, -sin], [sin, cos]])
        b = torch.Tensor([[(1 - cos) * cx + cy * sin], [(1 - cos) * cy - cx * sin]])
        new_points = M.mm( polygons ) + b
        new_points = new_points.t().reshape( (-1, 8) )
        xmins, _ = torch.min( new_points[:,  ::2], 1 )
        ymins, _ = torch.min( new_points[:, 1::2], 1 )
        xmaxs, _ = torch.max( new_points[:,  ::2], 1 )
        ymaxs, _ = torch.max( new_points[:, 1::2], 1 )
        boxes = torch.stack( [xmins, ymins, xmaxs, ymaxs], 1 ).reshape((-1, 4))

        new_target = BoxList( boxes, image.size, mode="xyxy" )
        new_target._copy_extra_fields( target )
        new_masks = SegmentationMask( new_points.reshape((-1, 1, 8)).tolist(), image.size, mode='poly' )
        new_target.add_field( "masks", new_masks )

        return new_image, new_target

@TRANSFORMS.register_module()
class RandomFlip:
    def __init__(self, prob=0.5,direction="horizontal"):
        assert direction in ['horizontal', 'vertical', 'diagonal'],f"{direction} not supported"
        self.direction = direction
        self.prob = prob

    def _flip_boxes(self,target,size):
        w,h = target["img_size"] 
        for key in ["bboxes","rboxes"]:
            if key not in target:
                continue
            bboxes = target[key]
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            target[key] = flipped

    def _flip_image(self,image):
        if self.direction=="horizontal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.direction == "vertical":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.direction == "diagonal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self._flip_image(image)
            if target is not None:
                self._flip_boxes(target)
        return image, target


class ToRect:
    def __init__( self, do=True ):
        self.do=do

    @staticmethod
    def _to_rrect( x ):
        x = cv2.minAreaRect( x )
        x = cv2.boxPoints( x )
        return x

    def __call__( self, image, target=None ):
        if target is None:
            return image

        if not self.do:
            return image, target

        masks = target.get_field( "masks" )
        polygons = list( map( lambda x: x.polygons[0].numpy(), masks.instances.polygons ) )
        polygons = np.stack( polygons, axis=0 ).reshape( (-1, 4, 2) )
        rrects = list( map( self._to_rrect, polygons ) )

        rrects_np = np.array( rrects, dtype=np.float32 ).reshape( (-1, 8) )
        xmins = np.min( rrects_np[:,  ::2], axis=1 )
        ymins = np.min( rrects_np[:, 1::2], axis=1 )
        xmaxs = np.max( rrects_np[:,  ::2], axis=1 )
        ymaxs = np.max( rrects_np[:, 1::2], axis=1 )
        xyxy = np.vstack( [xmins, ymins, xmaxs, ymaxs] ).transpose()
        boxes = torch.from_numpy( xyxy ).reshape(-1, 4)  # guard against no boxes

        new_target = BoxList( boxes, image.size, mode="xyxy" )
        new_target._copy_extra_fields( target )
        new_masks = SegmentationMask( rrects_np.reshape( (-1, 1, 8)).tolist(), image.size, mode='poly' )
        new_target.add_field( "masks", new_masks )

        return image, new_target

class SortForQuad( object ):
    def __init__( self, do=True ):
        self.do=do

    @staticmethod
    def choose_best_pointorder_fit_another(poly1):
        x1 = poly1[0]
        y1 = poly1[1]
        x2 = poly1[2]
        y2 = poly1[3]
        x3 = poly1[4]
        y3 = poly1[5]
        x4 = poly1[6]
        y4 = poly1[7]

        xmin = min( x1, x2, x3, x4 )
        ymin = min( y1, y2, y3, y4 )
        xmax = max( x1, x2, x3, x4 )
        ymax = max( y1, y2, y3, y4 )
        poly2 = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

        combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                     np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
        dst_coordinate = np.array(poly2)
        distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
        sorted = distances.argsort()
        return combinate[sorted[0]].tolist()

    def __call__( self, image, target=None ):
        if target == None:
            return image
        if not self.do:
            return image, target

        masks = target.get_field( "masks" )
        polygons = list( map( lambda x: x.polygons[0].numpy(), masks.instances.polygons ) )
        polygons = np.stack( polygons, axis=0 ).reshape( (-1, 8) )

        new_polygons = []
        for polygon in polygons:
            new_polygon = self.choose_best_pointorder_fit_another( polygon )
            new_polygons.append( [new_polygon] )

        new_masks = SegmentationMask( new_polygons, image.size, mode='poly' )
        target.add_field( "masks", new_masks )

        return image, target


class RandomCrop( object ):
    def __init__( self, size ):
        self.crop_size = size

    def __call__( self, image, target ):
        width, height = image.size
        i = random.choice( np.arange( 0, height - self.crop_size[0] ) )
        j = random.choice( np.arange( 0, width - self.crop_size[1] ) )
        image = F.crop( image, i, j, self.crop_size[0], self.crop_size[1] ) #image[i:i+width,j+height,:]
        target_ = target.crop( ( j, i, j + self.crop_size[1], i + self.crop_size[0]) )
        return image, target_

@TRANSFORMS.register_module()
class Normalize:
    def __init__(self, mean, std, to_bgr=True):
        self.mean = np.float32(mean).reshape(-1,1,1)
        self.std = np.float32(std).reshape(-1,1,1)
        self.to_bgr = to_bgr

    def __call__(self, image, target=None):
        if self.to_bgr:
            image = image[::-1]
        if isinstance(image, Image.Image):
            image = (np.array(image).transpose((2,0,1)) - self.mean*np.float32(255.)) * (np.float32(1./255.)/self.std)
        else:
            image = (image - self.mean) / self.std
        return image, target
