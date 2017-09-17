#!/usr/bin/env python2

from __future__ import print_function

import cv2
import numpy as np

import click


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


triangle_cache = {}


def draw_delaunay(img, visImg, subdiv, norm_type):
    triangleList = subdiv.getTriangleList()
    (height, width, _) = visImg.shape
    mask = np.zeros((height, width), np.uint8)

    max_dist = 0
    max_tri = None
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        (mean_color, dist) = (None, None)
        if (pt1, pt2, pt3) in triangle_cache:
            (mean_color, dist, poly) = triangle_cache[(pt1, pt2, pt3)]
            cv2.fillConvexPoly(visImg, poly, mean_color)
        else:
            poly = np.array([pt1, pt2, pt3], int)
            mask[...] = 0
            cv2.fillConvexPoly(mask, poly, 255)
            mean_color = cv2.mean(img, mask)
            cv2.fillConvexPoly(visImg, poly, mean_color)
            if not all([
                    rect_contains((0, 0, width, height), p)
                    for p in [pt1, pt2, pt3]
            ]):
                continue
            dist = cv2.norm(img, visImg, norm_type, mask)
            triangle_cache[(pt1, pt2, pt3)] = (mean_color, dist, poly)

        if dist > max_dist:
            max_dist = dist
            max_tri = (pt1, pt2, pt3)
    return (max_tri, max_dist)


def write_svg(img, subdiv, outfile):
    triangleList = subdiv.getTriangleList()
    (height, width, _) = img.shape
    mask = np.zeros((height, width), np.uint8)
    outfile.write('<svg width="{width}" height="{height}">\n'.format(
        width=width, height=height))
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if not all(
            [rect_contains((0, 0, width, height), p)
             for p in [pt1, pt2, pt3]]):
            continue
        poly = np.array([pt1, pt2, pt3], int)
        mask[...] = 0
        cv2.fillConvexPoly(mask, poly, 255)
        mean_color = cv2.mean(img, mask)
        color = "rgb({r}, {g}, {b})".format(
            r=int(mean_color[2]), g=int(mean_color[1]), b=int(mean_color[0]))
        outfile.write(
            '<polygon points="{t0},{t1} {t2},{t3} {t4},{t5}" style="fill:{color};stroke:{color}"/>\n'.
            format(
                t0=int(t[0]),
                t1=int(t[1]),
                t2=int(t[2]),
                t3=int(t[3]),
                t4=int(t[4]),
                t5=int(t[5]),
                color=color))
    outfile.write('</svg>\n')


@click.command()
@click.option('--interactive', is_flag=True, default=False)
@click.option('--until-distance', default=None, type=int)
@click.option('--max-iterations', default=None, type=int)
@click.option(
    '--distance-function',
    type=click.Choice(['inf', 'l1', 'l2', 'l2sqr']),
    default='l2')
@click.argument('image')
def cli(interactive, until_distance, max_iterations, distance_function, image):
    if (not until_distance) and (not max_iterations) and (not interactive):
        interactive = True

    norm_dist = {
        'inf': cv2.NORM_INF,
        'l1': cv2.NORM_L1,
        'l2': cv2.NORM_L2,
        'l2sqr': cv2.NORM_L2SQR,
    }[distance_function]

    img = cv2.imread(image)
    (height, width, _) = img.shape

    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    points = []
    cv2.namedWindow('vis')

    subdiv.insert((0, 0))
    subdiv.insert((width - 1, 0))
    subdiv.insert((width - 1, height - 1))
    subdiv.insert((0, height - 1))

    subdiv.insert((width / 2, 0))
    subdiv.insert((width - 1, height / 2))
    subdiv.insert((width / 2, height - 1))
    subdiv.insert((0, height / 2))

    show_points = False
    visImg = img.copy()
    (max_tri, max_dist) = draw_delaunay(img, visImg, subdiv, norm_dist)
    iteration = 0
    while True:
        if show_points:
            for p in points:
                cv2.circle(visImg, p, 3, (255, 255, 255))

        def _iterate(max_tri):
            m = cv2.moments(np.array(max_tri, np.int))
            mass_center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            points.append(mass_center)
            subdiv.insert(mass_center)
            visImg = img.copy()
            (max_tri, max_dist) = draw_delaunay(img, visImg, subdiv, norm_dist)
            print("new max_dist: %s" % max_dist)
            return (max_tri, max_dist, visImg)

        if interactive:
            cv2.imshow('vis', visImg)
            key = cv2.waitKey(20) & 0xff
            if key == 27:
                break
            if key == ord('p'):
                show_points = not show_points
            if (key == 32) or (until_distance and
                               (max_dist > until_distance)) or (
                                   iteration < max_iterations):
                (max_tri, max_dist, visImg) = _iterate(max_tri)
                iteration += 1
            if key == ord('s'):
                base_fname = '{origname}_iteration{iteration}'.format(
                    origname=image, iteration=iteration)
                cv2.imwrite(base_fname + '.png', visImg)
                with open(base_fname + '.svg', 'w') as outfile:
                    write_svg(img, subdiv, outfile)

        elif (until_distance and
              (max_dist > until_distance)) or (max_iterations and
                                               (iteration < max_iterations)):
            (max_tri, max_dist, visImg) = _iterate(max_tri)
            iteration += 1
        elif (until_distance and
              (max_dist <= until_distance)) or (max_iterations and
                                                (iteration >= max_iterations)):
            interactive = True
        else:
            key = cv2.waitKey(100) & 0xff

    cv2.destroyAllWindows()


if __name__ == '__main__':
    cli()
