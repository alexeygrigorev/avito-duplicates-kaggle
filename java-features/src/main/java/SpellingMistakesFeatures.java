import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.csv.CSVRecord;
import org.languagetool.JLanguageTool;
import org.languagetool.language.Russian;
import org.languagetool.rules.RuleMatch;

import com.aol.simple.react.stream.lazy.LazyReact;
import com.aol.simple.react.stream.traits.LazyFutureStream;
import com.fasterxml.jackson.jr.ob.JSON;
import com.google.common.base.Optional;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Maps;

public class SpellingMistakesFeatures {

    public static void main(String[] args) throws IOException {
        Iterable<CSVRecord> csv = Utils.itemInfoData();

        ThreadLocal<JLanguageTool> langTool = ThreadLocal.withInitial(() -> {
            return new JLanguageTool(new Russian());
        });

        int cores = Runtime.getRuntime().availableProcessors();
        LazyFutureStream<CSVRecord> stream = LazyReact.parallelBuilder(cores).from(csv.iterator());

        AtomicInteger cnt = new AtomicInteger();

        Stopwatch stopwatch = Stopwatch.createStarted();

        LazyFutureStream<String> misspellings = stream.map(rec -> {
            String id = rec.get("itemID");
            String title = rec.get("title");
            String description = rec.get("description");

            Map<String, Integer> titleMisspelling = extractMisspellingFeatures(langTool.get(), title);
            Map<String, Integer> descriptionMisspelling = extractMisspellingFeatures(langTool.get(), description);

            if (titleMisspelling.isEmpty() && descriptionMisspelling.isEmpty()) {
                return Optional.<String>absent();
            }

            Map<String, Object> map = new HashMap<>();
            map.put("_id", id);
            map.put("title", titleMisspelling);
            map.put("description", descriptionMisspelling);

            try {
                String json = JSON.std.asString(map);
                return Optional.of(json);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }).peek(s -> {
            int it = cnt.getAndIncrement();
            if (it % 100 == 0) {
                System.out.println("iteration number " + it);
            }
        }).filter(Optional::isPresent).map(Optional::get);

        PrintWriter pw = new PrintWriter("misspellings.json");
        misspellings.forEach(pw::println);
        pw.close();

        System.out.println("Computing spelling mistakes took " + stopwatch);

        ThreadPoolExecutor executor = (ThreadPoolExecutor) stream.getTaskExecutor();
        executor.shutdown();
    }


    private static Map<String, Integer> extractMisspellingFeatures(JLanguageTool langTool, String text) {
        try {
            Map<String, Integer> rules = Maps.newHashMap();

            List<RuleMatch> matches = langTool.check(text);

            for (RuleMatch match : matches) {
                String ruleName = match.getRule().getClass().getSimpleName();
                Integer value = rules.getOrDefault(ruleName, 0);
                rules.put(ruleName, value + 1);
            }

            return rules;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
