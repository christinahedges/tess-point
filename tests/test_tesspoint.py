from tesspoint import TESSPoint, footprint


def test_tesspoint():
    tp = TESSPoint(1, 1, 1)
    tp.pix2radec(footprint())

def test_conversion()
    tp = TESSPoint(1, 1, 1)
    footprint_radec = tp.pix2radec(footprint())
    footprint_convert = tp.radec2pix(footprint_radec)
    print(footprint()-footprint_convert)
