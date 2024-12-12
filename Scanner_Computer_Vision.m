%******************************************************************************************************************************
%
% Final Project: Paper Homography
% Computer Vision
% Fall 2024
% Authors:  Chanakya Nalapareddy and Nosherwan Babar
% Date: 12/11/2024
%
%******************************************************************************************************************************

% Clear workspace, command window, and close all figures
clear; clc; close all;

% Step 1: Read and preprocess the image
img = imread('IMG_1394.jpg'); % Replace with your file name

% Display the original image
figure;
imshow(img);
title('Original Image');

% Convert to grayscale if necessary using custom myRgb2Gray
img = myRgb2Gray(img);

% Resize the image while maintaining aspect ratio
[originalHeight, originalWidth] = size(img);
newHeight = 512;
newWidth = round((newHeight / originalHeight) * originalWidth);
resizedImg = imresize(img, [newHeight, newWidth]);

% Save the resized image in JPG format
imwrite(resizedImg, 'resized_image.jpg', 'jpg');

% Step 2: Manually apply Gaussian blur instead of imgaussfilt
sigma = 2; 
blurredImg = myGaussianFilter(resizedImg, sigma);

% Step 3: Perform adaptive edge detection
% Compute mean and std to set adaptive thresholds for Canny
meanVal = mean(blurredImg(:));
stdVal = std(double(blurredImg(:)));

lowThresh = max(0, (meanVal - stdVal) / 255);
highThresh = min(1, (meanVal + stdVal) / 255);
% Ensure thresholds are reasonable
lowThresh = max(0.1, min(lowThresh, 0.2));
highThresh = max(0.2, min(highThresh, 0.3));

edgeImg = edge(blurredImg, 'Canny', [lowThresh, highThresh]);

% Morphological closing to strengthen line segments
se1 = strel('line',5,0);
se2 = strel('line',5,90);
edgeImg = imclose(edgeImg, se1);
edgeImg = imclose(edgeImg, se2);

% Create a single window with 1 row and 3 columns of subplots
figure;

% Subplot 1: Grayscale Image
subplot(1, 3, 1);
imshow(resizedImg);
title('Original Grayscale Image');

% Subplot 2: Edge-Detected Image (after morph operations)
subplot(1, 3, 2);
imshow(edgeImg);
title('Edge-Detected Image');
hold off;

% Step 4: Find contours and filter by size
filledImg = myImfillHoles(edgeImg);
CC = myBwConnComp(filledImg);
stats = myRegionProps(CC);

if isempty(stats)
    warning('No connected regions found. Check your edge detection parameters.');
    return; 
end

% Identify the largest rectangular region (assume it's the paper)
maxArea = 0;
boundingBox = [];
for k = 1:length(stats)
    if stats(k).Area > maxArea
        maxArea = stats(k).Area;
        boundingBox = stats(k).BoundingBox;
    end
end

if isempty(boundingBox)
    warning('No large region found. The document may not be detected.');
    return;
end

% Create a mask for the detected paper region
paperMask = zeros(size(filledImg));
paperMask(round(boundingBox(2)):round(boundingBox(2) + boundingBox(4) - 1), ...
          round(boundingBox(1)):round(boundingBox(1) + boundingBox(3) - 1)) = 1;

filteredEdges = edgeImg & paperMask;

% Step 5: Perform Hough Transform on filtered edges
[H, theta, rho] = myHoughTransform(filteredEdges);

% Display the Hough Transform without marked peaks
figure;
imshow(imadjust(rescale(H)), [], 'XData', theta, 'YData', rho, ...
       'InitialMagnification', 'fit');
xlabel('\theta (degrees)');
ylabel('\rho (pixels)');
title('Hough Transform'); 
axis on;
axis normal;
hold off;

% Compute threshold for peaks
threshold = 0.2 * max(H(:));
peaks = myHoughPeaks(H, 50, threshold, [15, 15]);

if isempty(peaks)
    warning('No peaks found in Hough Transform. Cannot detect lines.');
    return;
end

% Adjusted parameters for myHoughLines if needed
FillGap = 30;    % increase gap fill to link more segments
MinLength = 40;  % ensure a bit longer lines for stability

lines = myHoughLines(filteredEdges, theta, rho, peaks, FillGap, MinLength);

[height, width] = size(filteredEdges);

% Check if lines were found
if isempty(lines)
    warning('No lines found. Check thresholds, morphological steps, or try adjusting parameters.');
    return;
end

% Combined visualization of Hough Transform with points and extended lines
figure;

% Subplot 1: Display the Hough Transform with marked points
subplot(1, 2, 1); 
imshow(imadjust(rescale(H)), [], 'XData', theta, 'YData', rho, ...
       'InitialMagnification', 'fit');
xlabel('\theta (degrees)');
ylabel('\rho (pixels)');
title('Hough Transform with Peaks');
axis on;
axis normal;
hold on;

plot(theta(peaks(:, 2)), rho(peaks(:, 1)), 'rs'); 
hold off;

% Subplot 2: Display the extended lines on the edge-detected image
subplot(1, 2, 2); 
imshow(filteredEdges);
hold on;

for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    if (xy(2, 1) - xy(1, 1)) ~= 0
        slope = (xy(2, 2) - xy(1, 2)) / (xy(2, 1) - xy(1, 1));
        intercept = xy(1, 2) - slope * xy(1, 1);
        
        x1 = 1; 
        y1 = slope * x1 + intercept;
        x2 = width; 
        y2 = slope * x2 + intercept;
        
        plot([x1, x2], [y1, y2], 'LineWidth', 1.0, 'Color', 'red');
    else
        x = xy(1, 1);
        plot([x, x], [1, height], 'LineWidth', 1.0, 'Color', 'red');
    end
end
title('Lines on Edge Image');
hold off;

% Step 6: Calculate intersections from unique extended lines
intersections = [];
for i = 1:length(lines)
    for j = i+1:length(lines)
        % Decrease angle threshold if needed
        if abs(lines(i).theta - lines(j).theta) < 5
            continue;
        end
        
        A = [cosd(lines(i).theta), sind(lines(i).theta);
             cosd(lines(j).theta), sind(lines(j).theta)];
        b = [lines(i).rho; lines(j).rho];
        if rank(A) == 2
            intersection = A \ b;
            if intersection(1) >= 1 && intersection(1) <= width && ...
               intersection(2) >= 1 && intersection(2) <= height
                intersections = [intersections; intersection'];
            end
        end
    end
end

if isempty(intersections)
    warning('No intersections found. Cannot identify corners.');
    return;
end

% Step 8: Closest point selection for corners
centerX = boundingBox(1) + boundingBox(3) / 2;
centerY = boundingBox(2) + boundingBox(4) / 2;
buffer = 20;

topLeft = intersections(intersections(:, 1) < centerX - buffer & intersections(:, 2) < centerY - buffer, :);
topRight = intersections(intersections(:, 1) >= centerX + buffer & intersections(:, 2) < centerY - buffer, :);
bottomLeft = intersections(intersections(:, 1) < centerX - buffer & intersections(:, 2) >= centerY + buffer, :);
bottomRight = intersections(intersections(:, 1) >= centerX + buffer & intersections(:, 2) >= centerY + buffer, :);

corners = [boundingBox(1), boundingBox(2); 
           boundingBox(1) + boundingBox(3), boundingBox(2); 
           boundingBox(1), boundingBox(2) + boundingBox(4); 
           boundingBox(1) + boundingBox(3), boundingBox(2) + boundingBox(4)];

consolidatedCorners = zeros(4, 2);

% If no suitable local corner is found, we pick the closest from global intersections
if ~isempty(topLeft)
    [~, idx] = min(vecnorm(topLeft - corners(1, :), 2, 2));
    consolidatedCorners(1, :) = topLeft(idx, :);
else
    [~, idx] = min(vecnorm(intersections - corners(1, :), 2, 2));
    consolidatedCorners(1, :) = intersections(idx, :);
end

if ~isempty(topRight)
    [~, idx] = min(vecnorm(topRight - corners(2, :), 2, 2));
    consolidatedCorners(2, :) = topRight(idx, :);
else
    [~, idx] = min(vecnorm(intersections - corners(2, :), 2, 2));
    consolidatedCorners(2, :) = intersections(idx, :);
end

if ~isempty(bottomLeft)
    [~, idx] = min(vecnorm(bottomLeft - corners(3, :), 2, 2));
    consolidatedCorners(3, :) = bottomLeft(idx, :);
else
    [~, idx] = min(vecnorm(intersections - corners(3, :), 2, 2));
    consolidatedCorners(3, :) = intersections(idx, :);
end

if ~isempty(bottomRight)
    [~, idx] = min(vecnorm(bottomRight - corners(4, :), 2, 2));
    consolidatedCorners(4, :) = bottomRight(idx, :);
else
    [~, idx] = min(vecnorm(intersections - corners(4, :), 2, 2));
    consolidatedCorners(4, :) = intersections(idx, :);
end

figure;
imshow(resizedImg);
hold on;
plot(consolidatedCorners(:, 1), consolidatedCorners(:, 2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('Corners Marked on Image');
hold off;

imwrite(resizedImg, 'final_filtered_detected_image.png');

% Step 10: Output the detected corners
disp('Detected Corners:');
disp(['Top-Left: [', num2str(consolidatedCorners(1, 1)), ', ', num2str(consolidatedCorners(1, 2)), ']']);
disp(['Top-Right: [', num2str(consolidatedCorners(2, 1)), ', ', num2str(consolidatedCorners(2, 2)), ']']);
disp(['Bottom-Right: [', num2str(consolidatedCorners(4, 1)), ', ', num2str(consolidatedCorners(4, 2)), ']']);
disp(['Bottom-Left: [', num2str(consolidatedCorners(3, 1)), ', ', num2str(consolidatedCorners(3, 2)), ']']);

orderedCorners = [consolidatedCorners(1, :); 
                  consolidatedCorners(2, :); 
                  consolidatedCorners(4, :); 
                  consolidatedCorners(3, :)];

disp('Ordered Corners:');
disp(orderedCorners);

imgPath = 'resized_image.jpg'; 
if ~isfile(imgPath)
    error('File not found. Please check the file path: %s', imgPath);
end
sourceImg = imread(imgPath);

sourceCorners = orderedCorners;

sourceWidth = max(sqrt(sum((sourceCorners(2, :) - sourceCorners(1, :)).^2)), ...
                  sqrt(sum((sourceCorners(3, :) - sourceCorners(4, :)).^2))); 
sourceHeight = max(sqrt(sum((sourceCorners(4, :) - sourceCorners(1, :)).^2)), ...
                   sqrt(sum((sourceCorners(3, :) - sourceCorners(2, :)).^2)));

targetWidth = ceil(sourceWidth);
targetHeight = ceil(sourceHeight);

targetCorners = [
    1, 1;
    targetWidth, 1;
    targetWidth, targetHeight;
    1, targetHeight
];

H = computeHomography(sourceCorners, targetCorners);
rectifiedImg = rectifyImageWithMapping(H, sourceImg, targetWidth, targetHeight);

figure;

% Display the grayscale image in the first subplot
subplot(1, 2, 1);
imshow(imgPath); 
title('Original Image');

% Display the rectified image in the second subplot
subplot(1, 2, 2);
imshow(rectifiedImg); 
title('Rectified Image');


%% Function to Compute the Homography Matrix
function H = computeHomography(sourceCorners, targetCorners)
    A = [];
    for i = 1:4
        x = sourceCorners(i, 1);
        y = sourceCorners(i, 2);
        u = targetCorners(i, 1);
        v = targetCorners(i, 2);

        A = [A;
             -x, -y, -1,  0,   0,   0, x*u, y*u, u;
              0,   0,   0, -x, -y, -1, x*v, y*v, v];
    end

    [~, ~, V] = svd(A);
    H = reshape(V(:, end), [3, 3])';
    H = H / H(3, 3);
end

%% Function to Perform Rectification using Backward Mapping
function rectifiedImg = rectifyImageWithMapping(H, sourceImg, targetWidth, targetHeight)
    H_inv = inv(H);
    rectifiedImg = zeros(targetHeight, targetWidth, size(sourceImg, 3), 'uint8');

    for v = 1:targetHeight
        for u = 1:targetWidth
            sourcePoint = H_inv * [u; v; 1];
            x = sourcePoint(1) / sourcePoint(3);
            y = sourcePoint(2) / sourcePoint(3);

            if x >= 1 && x <= size(sourceImg, 2) && y >= 1 && y <= size(sourceImg, 1)
                x1 = floor(x); x2 = ceil(x);
                y1 = floor(y); y2 = ceil(y);

                wx = x - x1;
                wy = y - y1;

                x1 = max(1, min(size(sourceImg, 2), x1));
                x2 = max(1, min(size(sourceImg, 2), x2));
                y1 = max(1, min(size(sourceImg, 1), y1));
                y2 = max(1, min(size(sourceImg, 1), y2));

                for c = 1:size(sourceImg, 3)
                    pixelValue = (1 - wx)*(1 - wy)*double(sourceImg(y1, x1, c)) + ...
                                 wx*(1 - wy)*double(sourceImg(y1, x2, c)) + ...
                                 (1 - wx)*wy*double(sourceImg(y2, x1, c)) + ...
                                 wx*wy*double(sourceImg(y2, x2, c));

                    rectifiedImg(v, u, c) = uint8(pixelValue);
                end
            end
        end
    end
end

%% Custom regionprops
function stats = myRegionProps(CC)
    stats = struct('Area', {}, 'BoundingBox', {});
    for i = 1:CC.NumObjects
        pixelIdx = CC.PixelIdxList{i};
        [rows, cols] = ind2sub(CC.ImageSize, pixelIdx);
        areaVal = length(pixelIdx);
        minRow = min(rows); maxRow = max(rows);
        minCol = min(cols); maxCol = max(cols);
        width = maxCol - minCol + 1;
        height = maxRow - minRow + 1;
        boundingBox = [minCol, minRow, width, height];
        
        stats(i).Area = areaVal;
        stats(i).BoundingBox = boundingBox;
    end
end

%% Custom Hough transform
function [H, theta, rho] = myHoughTransform(edgeImage)
    [height, width] = size(edgeImage);
    theta = -90:1:89;
    diagLen = ceil(sqrt(height^2 + width^2));
    rho = -diagLen:diagLen;
    H = zeros(length(rho), length(theta));
    [yCoords, xCoords] = find(edgeImage);
    for p = 1:length(xCoords)
        x = xCoords(p);
        y = yCoords(p);
        for tIdx = 1:length(theta)
            t = theta(tIdx);
            r = round(x*cosd(t) + y*sind(t));
            rIdx = r + diagLen + 1;
            H(rIdx, tIdx) = H(rIdx, tIdx) + 1;
        end
    end
end

%% Custom houghpeaks
function peaks = myHoughPeaks(H, numPeaks, threshold, nHoodSize)
    peaks = [];
    Htemp = H;
    halfNHood = floor((nHoodSize - 1)/2);
    for i = 1:numPeaks
        [val, idx] = max(Htemp(:));
        if val < threshold
            break;
        end
        [r, c] = ind2sub(size(H), idx);
        peaks = [peaks; r, c];
        
        rmin = max(r - halfNHood(1), 1);
        rmax = min(r + halfNHood(1), size(H,1));
        cmin = max(c - halfNHood(2), 1);
        cmax = min(c + halfNHood(2), size(H,2));
        
        Htemp(rmin:rmax, cmin:cmax) = 0;
    end
end

%% Custom Gaussian Filter Implementation (Manual conv2)
function output = myGaussianFilter(img, sigma)
    kernelSize = 2*ceil(3*sigma)+1; 
    halfSize = floor(kernelSize/2);
    [x, y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);
    gaussKernel = exp(-(x.^2 + y.^2)/(2*sigma^2));
    gaussKernel = gaussKernel / sum(gaussKernel(:));
    
    % Manual convolution instead of conv2
    convResult = myConvolve2D(double(img), gaussKernel);
    output = uint8(convResult);
end

%% Manual 2D Convolution Implementation
function convOut = myConvolve2D(inputImg, kernel)
    [h, w] = size(inputImg);
    [kh, kw] = size(kernel);
    padH = floor(kh/2);
    padW = floor(kw/2);
    
    padded = zeros(h+2*padH, w+2*padW);
    padded(padH+1:padH+h, padW+1:padW+w) = inputImg;
    
    convOut = zeros(h, w);
    for r = 1:h
        for c = 1:w
            region = padded(r:r+kh-1, c:c+kw-1);
            convOut(r,c) = sum(region(:).*kernel(:));
        end
    end
end

%% Custom imfill for holes
function filledImg = myImfillHoles(BW)
    invBW = ~BW;
    filledImg = BW; 
    [h, w] = size(BW);
    visited = false(h,w);
    queue = [];
    for x = 1:w
        if invBW(1,x) && ~visited(1,x)
            queue = [queue; 1, x];
            visited(1,x) = true;
        end
        if invBW(h,x) && ~visited(h,x)
            queue = [queue; h, x];
            visited(h,x) = true;
        end
    end
    for y = 1:h
        if invBW(y,1) && ~visited(y,1)
            queue = [queue; y, 1];
            visited(y,1) = true;
        end
        if invBW(y,w) && ~visited(y,w)
            queue = [queue; y, w];
            visited(y,w) = true;
        end
    end
    
    directions = [0 1;1 0;0 -1;-1 0];
    while ~isempty(queue)
        pt = queue(1,:);
        queue(1,:) = [];
        for d = 1:4
            ny = pt(1)+directions(d,1);
            nx = pt(2)+directions(d,2);
            if ny>=1 && ny<=h && nx>=1 && nx<=w
                if invBW(ny,nx) && ~visited(ny,nx)
                    visited(ny,nx) = true;
                    queue = [queue; ny, nx];
                end
            end
        end
    end
    
    holes = invBW & ~visited;
    filledImg(holes) = true;
end

%% Custom bwconncomp
function CC = myBwConnComp(BW)
    [h, w] = size(BW);
    labels = zeros(h,w,'uint32');
    labelCount = 0;
    directions = [0 1;1 0;0 -1;-1 0];
    PixelIdxList = {};
    for r = 1:h
        for c = 1:w
            if BW(r,c) && labels(r,c)==0
                labelCount = labelCount + 1;
                compPixels = [];
                queue = [r, c];
                labels(r,c) = labelCount;
                
                while ~isempty(queue)
                    pt = queue(1,:);
                    queue(1,:) = [];
                    compPixels = [compPixels; sub2ind([h w], pt(1), pt(2))];
                    
                    for d = 1:4
                        ny = pt(1)+directions(d,1);
                        nx = pt(2)+directions(d,2);
                        if ny>=1 && ny<=h && nx>=1 && nx<=w
                            if BW(ny,nx) && labels(ny,nx)==0
                                labels(ny,nx) = labelCount;
                                queue = [queue; ny, nx];
                            end
                        end
                    end
                end
                PixelIdxList{labelCount} = compPixels; %#ok<AGROW>
            end
        end
    end
    
    CC.NumObjects = labelCount;
    CC.PixelIdxList = PixelIdxList;
    CC.ImageSize = [h, w];
end

%% Custom rgb2gray
function grayImg = myRgb2Gray(img)
    if ndims(img) == 3
        img = double(img);
        R = img(:,:,1);
        G = img(:,:,2);
        B = img(:,:,3);
        gray = 0.2989 * R + 0.5870 * G + 0.1140 * B;
        grayImg = uint8(gray);
    else
        grayImg = img;
    end
end

%% Custom houghlines
function lines = myHoughLines(edgeImg, theta, rho, peaks, FillGap, MinLength)
    lines = struct('point1',{},'point2',{},'theta',{},'rho',{});
    [height, width] = size(edgeImg);
    [yCoords, xCoords] = find(edgeImg);
    points = [xCoords, yCoords];
    
    lineCount = 0;
    for p = 1:size(peaks,1)
        rInd = peaks(p,1);
        tInd = peaks(p,2);
        rVal = rho(rInd);
        tVal = theta(tInd);
        
        % Increase tol for more robust line detection
        tol = 3; 
        cosT = cosd(tVal);
        sinT = sind(tVal);
        
        dists = abs(points(:,1)*cosT + points(:,2)*sinT - rVal);
        linePoints = points(dists <= tol, :);
        
        if isempty(linePoints)
            continue;
        end
        
        if abs(sinT)>abs(cosT)
            [~, sortIdx] = sort(linePoints(:,2));
        else
            [~, sortIdx] = sort(linePoints(:,1));
        end
        linePoints = linePoints(sortIdx,:);
        
        segments = {};
        segStart = linePoints(1,:);
        prevPoint = segStart;
        
        for i = 2:size(linePoints,1)
            currPoint = linePoints(i,:);
            gap = sqrt((currPoint(1)-prevPoint(1))^2+(currPoint(2)-prevPoint(2))^2);
            if gap > FillGap
                segments{end+1} = [segStart; prevPoint]; %#ok<AGROW>
                segStart = currPoint;
            end
            prevPoint = currPoint;
        end
        segments{end+1} = [segStart; prevPoint]; %#ok<AGROW>
        
        for s = 1:length(segments)
            seg = segments{s};
            segLen = sqrt((seg(2,1)-seg(1,1))^2 + (seg(2,2)-seg(1,2))^2);
            if segLen >= MinLength
                lineCount = lineCount + 1;
                lines(lineCount).point1 = seg(1,:);
                lines(lineCount).point2 = seg(2,:);
                lines(lineCount).theta = tVal;
                lines(lineCount).rho = rVal;
            end
        end
    end
end

%************************THE-END***********************************************************************************************