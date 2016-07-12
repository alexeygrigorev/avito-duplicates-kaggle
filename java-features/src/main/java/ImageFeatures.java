import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang3.SerializationUtils;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.LocalFeatureMatcher;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.pixel.statistics.HistogramModel;
import org.openimaj.math.geometry.transforms.estimation.RobustAffineTransformEstimator;
import org.openimaj.math.model.fit.RANSAC;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;
import org.openimaj.util.pair.Pair;

import com.aol.simple.react.stream.lazy.LazyReact;
import com.aol.simple.react.stream.traits.LazyFutureStream;
import com.fasterxml.jackson.jr.ob.JSON;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Maps;

public class ImageFeatures {

    private static final String INPUT_FOLDER = Utils.inputFolder();
    private static final boolean UZE_ZIP_IMAGES = Utils.useZip();

    private static final ThreadLocal<DoGSIFTEngine> KEYPOINT_EXTRACTOR = ThreadLocal.withInitial(DoGSIFTEngine::new);
    private static final ThreadLocal<HistogramModel> HISTOGRAM_MODEL = ThreadLocal
            .withInitial(() -> new HistogramModel(10, 10, 10));
    private static final ThreadLocal<LocalFeatureMatcher<Keypoint>> FEATURE_MATCHER = ThreadLocal.withInitial(() -> {
        RobustAffineTransformEstimator fitter = new RobustAffineTransformEstimator(5.0, 50,
                new RANSAC.PercentageInliersStoppingCondition(0.5));

        FastBasicKeypointMatcher<Keypoint> fastMatcher = new FastBasicKeypointMatcher<>(8);
        ConsistentLocalFeatureMatcher2d<Keypoint> matcher = new ConsistentLocalFeatureMatcher2d<>(fastMatcher);
        matcher.setFittingModel(fitter);
        return matcher;
    });

    private static final List<DoubleFVComparison> HISTOGRAM_METRICS = Arrays.asList(DoubleFVComparison.SUM_SQUARE,
            DoubleFVComparison.CHI_SQUARE, DoubleFVComparison.COSINE_DIST, DoubleFVComparison.JACCARD_DISTANCE,
            DoubleFVComparison.HAMMING, DoubleFVComparison.CORRELATION, DoubleFVComparison.BHATTACHARYYA);

    public static void main(String[] args) throws Exception {
        Map<String, String[]> imageIds = readImageIds();
        Iterable<CSVRecord> itemPairs = Utils.itemPairsData();

        int cores = Runtime.getRuntime().availableProcessors();
        LazyFutureStream<CSVRecord> stream = LazyReact.parallelBuilder(cores).from(itemPairs.iterator());

        File outputFile = new File("image-features.json");
        PrintWriter pw;

        Set<String> processed = new HashSet<>();

        if (outputFile.exists()) {
            File cleanFile = new File("image-features-clean.json");
            PrintWriter clean = new PrintWriter(cleanFile);
            LineIterator li = FileUtils.lineIterator(outputFile);

            int cnt = 0;
            while (li.hasNext()) {
                try {
                    String line = li.next();
                    Map<Object, Object> map = JSON.std.mapFrom(line);
                    String id1 = map.get("ad_id_1").toString();
                    String id2 = map.get("ad_id_2").toString();
                    processed.add(id1 + "_" + id2);
                    clean.println(line);
                } catch (Exception e) {
                    System.out.print("Error while reading output at line " + cnt + ": ");
                    e.printStackTrace(System.out);
                }
                cnt++;
            }
            li.close();
            clean.close();

            System.out.println("Already processed " + processed.size() + " pairs. Continuing...");
            FileUtils.deleteQuietly(outputFile);
            FileUtils.moveFile(cleanFile, outputFile);

            pw = new PrintWriter(new BufferedOutputStream(new FileOutputStream(outputFile, true)));
            pw.println();
        } else {
            pw = new PrintWriter("image-features.json");
        }

        Stopwatch stopwatch = Stopwatch.createStarted();

        AtomicInteger cnt = new AtomicInteger(processed.size());
        AtomicInteger realCnt = new AtomicInteger(0);

        LazyFutureStream<String> imageFeatures = stream.filter(rec -> {
            String id1 = rec.get("itemID_1");
            String id2 = rec.get("itemID_2");
            return !processed.contains(id1 + "_" + id2);
        }).map(rec -> {
            String id1 = rec.get("itemID_1");
            String id2 = rec.get("itemID_2");

            String[] imagesIdsAd1 = imageIds.get(id1);
            Map<String, MBFImage> images1 = getImages(imagesIdsAd1);

            String[] imagesIdsAd2 = imageIds.get(id2);
            Map<String, MBFImage> images2 = getImages(imagesIdsAd2);

            Map<String, Object> features;
            if (images1.isEmpty() || images2.isEmpty()) {
                features = new HashMap<>();
            } else {
                features = calculateFeatures(images1, images2);
            }

            features.put("ad_id_1", id1);
            features.put("ad_id_2", id2);

            try {
                return JSON.std.asString(features);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }).peek(s -> {
            int it = cnt.incrementAndGet();
            realCnt.incrementAndGet();
            if (it % 100 == 0) {
                System.out.print("iteration number " + it + ". ");
                long elapsed = stopwatch.elapsed(TimeUnit.SECONDS);
                System.out.print(realCnt.get() * 1.0 / elapsed);
                System.out.println(" iterations per second");
            }
        });

        imageFeatures.forEach(pw::println);
        pw.close();

        System.out.println("computing image features took " + stopwatch.stop());
        long elapsed = stopwatch.elapsed(TimeUnit.SECONDS);
        System.out.print(cnt.get() * 1.0 / elapsed);
        System.out.println(" iterations per second");

        ThreadPoolExecutor executor = (ThreadPoolExecutor) stream.getTaskExecutor();
        executor.shutdown();

    }

    private static Map<String, Object> calculateFeatures(Map<String, MBFImage> images1, Map<String, MBFImage> images2) {
        Map<String, Object> features = new HashMap<>();
        siftFeatutes(features, images1, images2);
        histogramFeatures(features, images1, images2);
        return features;
    }

    private static void siftFeatutes(Map<String, Object> features, Map<String, MBFImage> images1,
            Map<String, MBFImage> images2) {

        Map<String, LocalFeatureList<Keypoint>> keypoints1 = keypoints(images1);
        for (Entry<String, LocalFeatureList<Keypoint>> e : keypoints1.entrySet()) {
            LocalFeatureList<Keypoint> image1 = e.getValue();
            features.put("ad1_" + e.getKey() + "_kp_no", image1.size());
        }

        Map<String, LocalFeatureList<Keypoint>> keypoints2 = keypoints(images2);
        for (Entry<String, LocalFeatureList<Keypoint>> e : keypoints2.entrySet()) {
            LocalFeatureList<Keypoint> image2 = e.getValue();
            features.put("ad2_" + e.getKey() + "_kp_no", image2.size());
        }

        LocalFeatureMatcher<Keypoint> matcher = FEATURE_MATCHER.get();

        for (Entry<String, LocalFeatureList<Keypoint>> e1 : keypoints1.entrySet()) {
            LocalFeatureList<Keypoint> image1 = e1.getValue();
            for (Entry<String, LocalFeatureList<Keypoint>> e2 : keypoints2.entrySet()) {
                String key1 = e1.getKey();
                String key2 = e2.getKey();

                LocalFeatureList<Keypoint> image2 = e2.getValue();

                matcher.setModelFeatures(image1);
                matcher.findMatches(image2);
                List<Pair<Keypoint>> matches = matcher.getMatches();
                // display(images1, images2, e1.getKey(), e2.getKey(), matches);
                features.put("keypoints_1_2_" + key1 + "_" + key2, matches.size());

                matcher.setModelFeatures(image2);
                matcher.findMatches(image1);
                matches = matcher.getMatches();
                features.put("keypoints_2_1_" + key1 + "_" + key2, matches.size());
                // display(images2, images1, e2.getKey(), e1.getKey(), matches);
            }
        }

    }

    private static Map<String, LocalFeatureList<Keypoint>> keypoints(Map<String, MBFImage> images) {
        DoGSIFTEngine eng = KEYPOINT_EXTRACTOR.get();

        Map<String, LocalFeatureList<Keypoint>> result = new HashMap<>();

        for (Entry<String, MBFImage> e : images.entrySet()) {
            FImage fImage = e.getValue().flatten();
            LocalFeatureList<Keypoint> keypoints = eng.findFeatures(fImage);
            result.put(e.getKey(), keypoints);
        }

        return result;
    }

    private static void histogramFeatures(Map<String, Object> features, Map<String, MBFImage> images1,
            Map<String, MBFImage> images2) {
        Map<String, MultidimensionalHistogram> hists1 = histograms(images1);
        Map<String, MultidimensionalHistogram> hists2 = histograms(images2);

        for (Entry<String, MultidimensionalHistogram> e1 : hists1.entrySet()) {
            MultidimensionalHistogram h1 = e1.getValue();
            for (Entry<String, MultidimensionalHistogram> e2 : hists2.entrySet()) {
                String key = "hist_" + e1.getKey() + "_" + e2.getKey();

                MultidimensionalHistogram h2 = e2.getValue();
                for (DoubleFVComparison metric : HISTOGRAM_METRICS) {
                    double result = h1.compare(h2, metric);
                    features.put(key + "_" + metric.name(), result);
                }
            }
        }
    }

    private static Map<String, MultidimensionalHistogram> histograms(Map<String, MBFImage> images) {
        HistogramModel histogramModel = HISTOGRAM_MODEL.get();
        Map<String, MultidimensionalHistogram> hist = new HashMap<>();

        for (Entry<String, MBFImage> e : images.entrySet()) {
            String key = e.getKey();
            MBFImage value = e.getValue();
            histogramModel.estimateModel(value);
            hist.put(key, histogramModel.histogram.clone());
        }

        return hist;
    }

    private static Map<String, MBFImage> getImages(String[] imageIds) {
        Map<String, MBFImage> result = new HashMap<>();

        for (String im : imageIds) {
            if (im.isEmpty()) {
                continue;
            }

            try {
                MBFImage mbfImage = findImage(im, UZE_ZIP_IMAGES);
                result.put(im, mbfImage);
            } catch (Exception e) {
                System.out.println(im + " not found: " + e.getMessage());
            }
        }

        return result;
    }

    private static MBFImage findImage(String imageId, boolean fromZip) throws Exception {
        String archive = "Images_" + imageId.charAt(imageId.length() - 2);
        int folderId = Integer.parseInt(imageId.substring(imageId.length() - 2));
        String name = archive + "/" + folderId + "/" + imageId + ".jpg";

        if (!fromZip) {
            String fullPath = INPUT_FOLDER + "/" + name;
            try (InputStream is = new BufferedInputStream(new FileInputStream(fullPath))) {
                return ImageUtilities.readMBF(is);
            }
        }

        File zipFile = new File(INPUT_FOLDER, archive + ".zip");
        try (ZipFile zip = new ZipFile(zipFile)) {
            ZipEntry entry = zip.getEntry(name);
            try (InputStream is = zip.getInputStream(entry)) {
                return ImageUtilities.readMBF(is);
            }
        }
    }

    private static Map<String, String[]> readImageIds() throws IOException {
        File file = new File("image_id.db");
        if (file.exists()) {
            return loadIds(file);
        }

        Stopwatch stopwatch = Stopwatch.createStarted();
        Iterable<CSVRecord> itemInfo = Utils.itemInfoData();
        Map<String, String[]> adImages = Maps.newHashMap();
        int cnt = 0;
        for (CSVRecord rec : itemInfo) {
            String images = rec.get("images_array");
            String[] split = images.split(", ");
            adImages.put(rec.get("itemID"), split);

            cnt++;
            if (cnt % 100000 == 0) {
                System.out.println("read " + cnt + " image ids...");
            }
        }

        System.out.println("reading image ids took " + stopwatch.stop());
        saveIds(file, adImages);
        return adImages;
    }

    private static Map<String, String[]> loadIds(File file) throws FileNotFoundException {
        Stopwatch stopwatch = Stopwatch.createStarted();
        try {
            BufferedInputStream is = new BufferedInputStream(new FileInputStream(file));
            return SerializationUtils.deserialize(is);
        } finally {
            System.out.println("loading serialized data took " + stopwatch.stop());
        }
    }

    private static void saveIds(File file, Map<String, String[]> adImages) throws FileNotFoundException {
        Stopwatch stopwatch = Stopwatch.createStarted();
        OutputStream outputStream = new BufferedOutputStream(new FileOutputStream(file));
        SerializationUtils.serialize((Serializable) adImages, outputStream);
        System.out.println("saving data took " + stopwatch.stop());
    }

}
