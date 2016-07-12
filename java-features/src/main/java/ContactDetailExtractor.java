
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.IOUtils;
import org.nibor.autolink.LinkExtractor;
import org.nibor.autolink.LinkSpan;
import org.nibor.autolink.LinkType;

import com.aol.simple.react.stream.lazy.LazyReact;
import com.aol.simple.react.stream.traits.LazyFutureStream;
import com.fasterxml.jackson.jr.ob.JSON;
import com.google.common.base.CharMatcher;
import com.google.common.base.Optional;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.i18n.phonenumbers.PhoneNumberMatch;
import com.google.i18n.phonenumbers.PhoneNumberUtil;
import com.google.i18n.phonenumbers.Phonenumber.PhoneNumber;
import com.twitter.Extractor;

public class ContactDetailExtractor {
    private static final Locale RUS = Locale.forLanguageTag("RU");

    private static final PhoneNumberUtil PHONE_EXTRACTOR = PhoneNumberUtil.getInstance();
    private static final LinkExtractor EMAIL_LINK_EXTRACTOR = LinkExtractor.builder()
            .linkTypes(EnumSet.of(LinkType.EMAIL)).build();

    private static final Set<String> GOOD_URL_ENDINGS = readGoodUrlEndings();
    private static final CharMatcher DIGITS = CharMatcher.DIGIT;
    private static final String DOT = Pattern.quote(".");
    private static final Pattern PLACEHOLDERS_PATTERN = Pattern.compile("@@.+?@@");

    public static List<String> extractLinks(String line) {
        line = line.replace("./", "/").replace("/.", "/");
        Extractor extractor = new Extractor();
        List<Extractor.Entity> urls = extractor.extractURLsWithIndices(line);

        List<String> result = Lists.newArrayList();
        for (Extractor.Entity match : urls) {
            String value = match.getValue();
            if (value.length() <= 5) {
                continue;
            }

            URL url;
            try {
                url = toUrl(value);
            } catch (MalformedURLException e) {
                continue;
            }

            String host = url.getHost();

            String[] split = host.split(DOT);
            if (split.length > 3) {
                continue;
            }

            String last = split[split.length - 1];
            if (!GOOD_URL_ENDINGS.contains(last)) {
                continue;
            }

            String domain = split[split.length - 2];
            if (DIGITS.matches(domain.charAt(0))) {
                continue;
            }

            if (value.endsWith("/")) {
                value = value.substring(0, value.length() - 1);
            }

            if (value.startsWith("http://")) {
                value = value.substring("http://".length());
            } else if (value.startsWith("https://")) {
                value = value.substring("https://".length());
            }
            if (value.startsWith("www.")) {
                value = value.substring("www.".length());
            }

            result.add(value);
        }

        return result;
    }

    private static URL toUrl(String value) throws MalformedURLException {
        if (!value.startsWith("http")) {
            value = "http://" + value;
        }

        return new URL(value);
    }

    public static List<String> extractEmails(String line) {
        line = line.replace("маил", "mail").replace("майл", "mail").replace("яндекс", "yandex");
        Iterable<LinkSpan> emails = EMAIL_LINK_EXTRACTOR.extractLinks(line);

        List<String> result = Lists.newArrayList();
        for (LinkSpan match : emails) {
            String email = line.substring(match.getBeginIndex(), match.getEndIndex());
            String[] split = email.split(DOT);
            String last = split[split.length - 1];
            if (!GOOD_URL_ENDINGS.contains(last)) {
                continue;
            }

            result.add(new String(email));
        }

        return result;
    }

    public static List<String> extractPhones(String line) {
        line = line.replaceAll("[.,!@#$%&*|_\t~]", " ")
                .replaceAll("дес[ия]ть?", "10")
                .replaceAll("[оа]динн?ад?цать?", "11")
                .replaceAll("дв[ие]над?цать?", "12")
                .replaceAll("тр[еи]над?цать?", "13")
                .replaceAll("ч[ие]тырнад?цать?", "14")
                .replaceAll("п[яе]ть?над?цать?", "15")
                .replaceAll("ш[ыие]сть?над?цать?", "16")
                .replaceAll("с[ие]мь?над?цать?", "17")
                .replaceAll("вос[ие]мь?над?цать?", "18")
                .replaceAll("дев[яе]ть?над?цать?", "19")
                .replaceAll("двад?ц?цать?", "2")
                .replaceAll("трид?ц?цать?", "3")
                .replaceAll("сор[оа]к", "4")
                .replaceAll("пять?д[ие]сять?", "5")
                .replaceAll("ш[ыие]сть?д[ие]сять?", "6")
                .replaceAll("семь?д[ие]сять?", "7")
                .replaceAll("вос[ие]мь?д[ие]сять?", "8")
                .replaceAll("д[ие]в[ея]носто", "9")
                .replaceAll("сто", "1")
                .replaceAll("двест[eи]", "2")
                .replaceAll("трист[ао]", "3")
                .replaceAll("ч[ие]тыр[ие]ст[ао]", "4")
                .replaceAll("п[ия]ть?сот", "5")
                .replaceAll("п[ия]цц?от", "5")
                .replaceAll("шес?т?ь?сот", "6")
                .replaceAll("с[еи]мь?сот", "7")
                .replaceAll("вос[еи]мь?сот", "8")
                .replaceAll("дев[яе]ть?сот", "9")
                .replaceAll("дев[яе]цц?от", "9")
                .replaceAll("[ао]дин", "1")
                .replaceAll("два", "2")
                .replaceAll("три", "3")
                .replaceAll("ч[ие]т[иы]р[ие]", "4")
                .replaceAll("пять", "5")
                .replaceAll("шесть?", "6")
                .replaceAll("семь?", "7")
                .replaceAll("вос[еи]мь?", "8")
                .replaceAll("н[оу]ль?", "0")
                .replaceAll("дев[ия]ть?", "9");

        Iterable<PhoneNumberMatch> numbers = PHONE_EXTRACTOR.findNumbers(line, "RU");
        List<String> result = Lists.newArrayList();
        for (PhoneNumberMatch match : numbers) {
            PhoneNumber number = match.number();
            result.add("+7" + number.getNationalNumber());
        }

        return result;
    }

    public static List<String> extractHashTags(String line) {
        Extractor extractor = new Extractor();
        List<Extractor.Entity> urls = extractor.extractHashtagsWithIndices(line);

        List<String> result = Lists.newArrayList();
        for (Extractor.Entity match : urls) {
            String value = match.getValue();
            result.add("#" + value);
        }

        return result;
    }

    public static List<String> extractPlaceholders(String line) {
        Matcher matcher = PLACEHOLDERS_PATTERN.matcher(line);
        List<String> result = Lists.newArrayList();
        while (matcher.find()) {
            String group = matcher.group(0);
            result.add(group);
        }

        return result;
    }

    private static void run() throws IOException, FileNotFoundException {
        Iterable<CSVRecord> csv = Utils.itemInfoData();

        int cores = Runtime.getRuntime().availableProcessors();
        LazyFutureStream<CSVRecord> stream = LazyReact.parallelBuilder(cores).from(csv.iterator());

        AtomicInteger cnt = new AtomicInteger();

        Stopwatch stopwatch = Stopwatch.createStarted();

        LazyFutureStream<String> misspellings = stream.map(rec -> {
            String id = rec.get("itemID");
            String title = rec.get("title");
            String description = rec.get("description");
            String allText = title + " " + description;
            allText = allText.toLowerCase(RUS);
            allText = allText.replace(".комп", ". комп").replace(".руб", ". руб").replace(".ру", ".ru")
                    .replace(".ком", ".com").replace(".орг", ".org");

            List<String> emails = extractEmails(allText);
            List<String> links = extractLinks(allText);
            List<String> phones = extractPhones(allText);
            List<String> hashtags = extractHashTags(allText);
            List<String> placeholders = extractPlaceholders(allText);

            if (emails.isEmpty() && links.isEmpty() && phones.isEmpty() && hashtags.isEmpty()
                    && placeholders.isEmpty()) {
                return Optional.<String> absent();
            }

            Map<String, Object> map = new HashMap<>();
            map.put("_id", id);
            if (!emails.isEmpty()) {
                map.put("emails", emails);
            }
            if (!links.isEmpty()) {
                map.put("links", links);
            }
            if (!phones.isEmpty()) {
                map.put("phones", phones);
            }
            if (!hashtags.isEmpty()) {
                map.put("hashtags", hashtags);
            }
            if (!placeholders.isEmpty()) {
                map.put("placeholders", placeholders);
            }

            try {
                String json = JSON.std.asString(map);
                return Optional.of(json);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }).peek(s -> {
            int it = cnt.getAndIncrement();
            if (it % 1000 == 0) {
                System.out.println("iteration number " + it);
            }
        }).filter(Optional::isPresent).map(Optional::get);

        PrintWriter pw = new PrintWriter("contacts.json");
        misspellings.forEach(pw::println);
        pw.close();

        System.out.println("Computing contact features mistakes took " + stopwatch);

        ThreadPoolExecutor executor = (ThreadPoolExecutor) stream.getTaskExecutor();
        executor.shutdown();
    }

    public static void main(String[] args) throws IOException {
        run();
    }

    private static Set<String> readGoodUrlEndings() {
        try {
            InputStream is = ContactDetailExtractor.class.getResourceAsStream("url-endings.txt");
            List<String> lines = IOUtils.readLines(is);
            return ImmutableSet.copyOf(lines);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}