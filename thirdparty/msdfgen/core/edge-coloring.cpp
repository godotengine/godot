
#include "edge-coloring.h"

namespace msdfgen {

static bool isCorner(const Vector2 &aDir, const Vector2 &bDir, double crossThreshold) {
    return dotProduct(aDir, bDir) <= 0 || fabs(crossProduct(aDir, bDir)) > crossThreshold;
}

static double estimateEdgeLength(const EdgeSegment *edge) {
    double len = 0;
    Point2 prev = edge->point(0);
    for (int i = 1; i <= MSDFGEN_EDGE_LENGTH_PRECISION; ++i) {
        Point2 cur = edge->point(1./MSDFGEN_EDGE_LENGTH_PRECISION*i);
        len += (cur-prev).length();
        prev = cur;
    }
    return len;
}

static void switchColor(EdgeColor &color, unsigned long long &seed, EdgeColor banned = BLACK) {
    EdgeColor combined = EdgeColor(color&banned);
    if (combined == RED || combined == GREEN || combined == BLUE) {
        color = EdgeColor(combined^WHITE);
        return;
    }
    if (color == BLACK || color == WHITE) {
        static const EdgeColor start[3] = { CYAN, MAGENTA, YELLOW };
        color = start[seed%3];
        seed /= 3;
        return;
    }
    int shifted = color<<(1+(seed&1));
    color = EdgeColor((shifted|shifted>>3)&WHITE);
    seed >>= 1;
}

void edgeColoringSimple(Shape &shape, double angleThreshold, unsigned long long seed) {
    double crossThreshold = sin(angleThreshold);
    std::vector<int> corners;
    for (std::vector<Contour>::iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
        // Identify corners
        corners.clear();
        if (!contour->edges.empty()) {
            Vector2 prevDirection = contour->edges.back()->direction(1);
            int index = 0;
            for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge, ++index) {
                if (isCorner(prevDirection.normalize(), (*edge)->direction(0).normalize(), crossThreshold))
                    corners.push_back(index);
                prevDirection = (*edge)->direction(1);
            }
        }

        // Smooth contour
        if (corners.empty())
            for (std::vector<EdgeHolder>::iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge)
                (*edge)->color = WHITE;
        // "Teardrop" case
        else if (corners.size() == 1) {
            EdgeColor colors[3] = { WHITE, WHITE };
            switchColor(colors[0], seed);
            switchColor(colors[2] = colors[0], seed);
            int corner = corners[0];
            if (contour->edges.size() >= 3) {
                int m = (int) contour->edges.size();
                for (int i = 0; i < m; ++i)
                    contour->edges[(corner+i)%m]->color = (colors+1)[int(3+2.875*i/(m-1)-1.4375+.5)-3];
            } else if (contour->edges.size() >= 1) {
                // Less than three edge segments for three colors => edges must be split
                EdgeSegment *parts[7] = { };
                contour->edges[0]->splitInThirds(parts[0+3*corner], parts[1+3*corner], parts[2+3*corner]);
                if (contour->edges.size() >= 2) {
                    contour->edges[1]->splitInThirds(parts[3-3*corner], parts[4-3*corner], parts[5-3*corner]);
                    parts[0]->color = parts[1]->color = colors[0];
                    parts[2]->color = parts[3]->color = colors[1];
                    parts[4]->color = parts[5]->color = colors[2];
                } else {
                    parts[0]->color = colors[0];
                    parts[1]->color = colors[1];
                    parts[2]->color = colors[2];
                }
                contour->edges.clear();
                for (int i = 0; parts[i]; ++i)
                    contour->edges.push_back(EdgeHolder(parts[i]));
            }
        }
        // Multiple corners
        else {
            int cornerCount = (int) corners.size();
            int spline = 0;
            int start = corners[0];
            int m = (int) contour->edges.size();
            EdgeColor color = WHITE;
            switchColor(color, seed);
            EdgeColor initialColor = color;
            for (int i = 0; i < m; ++i) {
                int index = (start+i)%m;
                if (spline+1 < cornerCount && corners[spline+1] == index) {
                    ++spline;
                    switchColor(color, seed, EdgeColor((spline == cornerCount-1)*initialColor));
                }
                contour->edges[index]->color = color;
            }
        }
    }
}

struct EdgeColoringInkTrapCorner {
    int index;
    double prevEdgeLengthEstimate;
    bool minor;
    EdgeColor color;
};

void edgeColoringInkTrap(Shape &shape, double angleThreshold, unsigned long long seed) {
    typedef EdgeColoringInkTrapCorner Corner;
    double crossThreshold = sin(angleThreshold);
    std::vector<Corner> corners;
    for (std::vector<Contour>::iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
        // Identify corners
        double splineLength = 0;
        corners.clear();
        if (!contour->edges.empty()) {
            Vector2 prevDirection = contour->edges.back()->direction(1);
            int index = 0;
            for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge, ++index) {
                if (isCorner(prevDirection.normalize(), (*edge)->direction(0).normalize(), crossThreshold)) {
                    Corner corner = { index, splineLength };
                    corners.push_back(corner);
                    splineLength = 0;
                }
                splineLength += estimateEdgeLength(*edge);
                prevDirection = (*edge)->direction(1);
            }
        }

        // Smooth contour
        if (corners.empty())
            for (std::vector<EdgeHolder>::iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge)
                (*edge)->color = WHITE;
        // "Teardrop" case
        else if (corners.size() == 1) {
            EdgeColor colors[3] = { WHITE, WHITE };
            switchColor(colors[0], seed);
            switchColor(colors[2] = colors[0], seed);
            int corner = corners[0].index;
            if (contour->edges.size() >= 3) {
                int m = (int) contour->edges.size();
                for (int i = 0; i < m; ++i)
                    contour->edges[(corner+i)%m]->color = (colors+1)[int(3+2.875*i/(m-1)-1.4375+.5)-3];
            } else if (contour->edges.size() >= 1) {
                // Less than three edge segments for three colors => edges must be split
                EdgeSegment *parts[7] = { };
                contour->edges[0]->splitInThirds(parts[0+3*corner], parts[1+3*corner], parts[2+3*corner]);
                if (contour->edges.size() >= 2) {
                    contour->edges[1]->splitInThirds(parts[3-3*corner], parts[4-3*corner], parts[5-3*corner]);
                    parts[0]->color = parts[1]->color = colors[0];
                    parts[2]->color = parts[3]->color = colors[1];
                    parts[4]->color = parts[5]->color = colors[2];
                } else {
                    parts[0]->color = colors[0];
                    parts[1]->color = colors[1];
                    parts[2]->color = colors[2];
                }
                contour->edges.clear();
                for (int i = 0; parts[i]; ++i)
                    contour->edges.push_back(EdgeHolder(parts[i]));
            }
        }
        // Multiple corners
        else {
            int cornerCount = (int) corners.size();
            int majorCornerCount = cornerCount;
            if (cornerCount > 3) {
                corners.begin()->prevEdgeLengthEstimate += splineLength;
                for (int i = 0; i < cornerCount; ++i) {
                    if (
                        corners[i].prevEdgeLengthEstimate > corners[(i+1)%cornerCount].prevEdgeLengthEstimate &&
                        corners[(i+1)%cornerCount].prevEdgeLengthEstimate < corners[(i+2)%cornerCount].prevEdgeLengthEstimate
                    ) {
                        corners[i].minor = true;
                        --majorCornerCount;
                    }
                }
            }
            EdgeColor color = WHITE;
            EdgeColor initialColor = BLACK;
            for (int i = 0; i < cornerCount; ++i) {
                if (!corners[i].minor) {
                    --majorCornerCount;
                    switchColor(color, seed, EdgeColor(!majorCornerCount*initialColor));
                    corners[i].color = color;
                    if (!initialColor)
                        initialColor = color;
                }
            }
            for (int i = 0; i < cornerCount; ++i) {
                if (corners[i].minor) {
                    EdgeColor nextColor = corners[(i+1)%cornerCount].color;
                    corners[i].color = EdgeColor((color&nextColor)^WHITE);
                } else
                    color = corners[i].color;
            }
            int spline = 0;
            int start = corners[0].index;
            color = corners[0].color;
            int m = (int) contour->edges.size();
            for (int i = 0; i < m; ++i) {
                int index = (start+i)%m;
                if (spline+1 < cornerCount && corners[spline+1].index == index)
                    color = corners[++spline].color;
                contour->edges[index]->color = color;
            }
        }
    }
}

}
