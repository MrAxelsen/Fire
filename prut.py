

class boundingBox:
    def __init__(self, ident, x, y, w, h):
        self.ident = ident
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def __repr__(self):
        return '<bbox x: %s, y: %s>' % (self.x, self.y)
    def __str__(self):
        return 'BoundingBox @ x: %s, y: %s' % (self.x, self.y)


def makebbox():
    bboxes = []
    for i in range(2):
        bbox = boundingBox(1,2,3,4,5)
        print(bbox)
        bboxes.append(bbox)
    return bboxes


bboxes = []
bb = makebbox()

bboxes.extend(bb)

for bbox in bboxes:
    bbox.x


