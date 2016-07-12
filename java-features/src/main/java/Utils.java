import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.zip.ZipFile;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.openimaj.feature.local.matcher.MatchingUtilities;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.math.geometry.point.Point2d;
import org.openimaj.math.geometry.point.Point2dImpl;
import org.openimaj.util.pair.Pair;

import com.google.common.base.Charsets;
import com.google.common.collect.Iterables;

public class Utils {

    public static Iterable<CSVRecord> itemInfoData() throws IOException {
        Iterable<CSVRecord> trainCsv = readCsv("ItemInfo_train.csv");
        Iterable<CSVRecord> testCsv = readCsv("ItemInfo_test.csv");
        return Iterables.concat(trainCsv, testCsv);
    }

    public static Iterable<CSVRecord> itemPairsData() throws IOException {
        Iterable<CSVRecord> trainCsv = readCsv("ItemPairs_train.csv");
        Iterable<CSVRecord> testCsv = readCsv("ItemPairs_test.csv");
        return Iterables.concat(trainCsv, testCsv);
    }

    private static Iterable<CSVRecord> readCsv(String fileName) throws IOException {
        InputStream is = readFile(fileName);
        CSVFormat reader = CSVFormat.RFC4180.withHeader();
        return reader.parse(new InputStreamReader(is, Charsets.UTF_8));
    }

    @SuppressWarnings("resource")
    public static InputStream readFile(String fileName) throws IOException {
        String fullPathZip = inputFolder() + "/" + fileName + ".zip";
        ZipFile file = new ZipFile(fullPathZip);
        return file.getInputStream(file.getEntry(fileName));
    }

    public static String inputFolder() {
        try {
            Properties properties = new Properties();
            properties.load(new FileReader("java-features.properties"));
            return properties.getProperty("input.folder");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static boolean useZip() {
        try {
            Properties properties = new Properties();
            properties.load(new FileReader("java-features.properties"));
            String property = properties.getProperty("use.zip");
            return "true".equals(property);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void display(Map<String, MBFImage> images1, Map<String, MBFImage> images2, String key1, String key2,
            List<Pair<Keypoint>> matches) {
        MBFImage im1 = images1.get(key1);
        MBFImage im2 = images2.get(key2);
        display(im1, im2, matches);
    }

    public static void display(MBFImage im1, MBFImage im2, List<Pair<Keypoint>> matches) {
        List<Pair<Point2d>> matches1 = covertMatchesToPoints(matches);
        MBFImage drawMatches = MatchingUtilities.drawMatches(im1, im2, matches1, RGBColour.RED);
        DisplayUtilities.display(drawMatches);
    }

    private static List<Pair<Point2d>> covertMatchesToPoints(List<Pair<Keypoint>> matches) {
        List<Pair<Point2d>> res = new ArrayList<>();
        for (Pair<Keypoint> kp : matches) {
            Keypoint first = kp.firstObject();
            Keypoint second = kp.secondObject();

            Point2dImpl p1 = new Point2dImpl(first.x, first.y);
            Point2dImpl p2 = new Point2dImpl(second.x, second.y);
            res.add(new Pair<>(p1, p2));
        }
        return res;
    }

}
