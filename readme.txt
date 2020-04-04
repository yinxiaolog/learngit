Git is a version control system.
Git is free software.
yinxiaolong
Git is a distributed version control system.
Git is free software distributed under the GPL.
Git has a mutable index called stage.
Creating a new brach is quick.
Create a new branch is quick & simple.
Create a new branch is quick AND simple.

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ReducingState;
import org.apache.flink.api.common.state.ReducingStateDescriptor;
import org.apache.flink.api.common.typeutils.base.LongSerializer;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.triggers.ContinuousProcessingTimeTrigger;
import org.apache.flink.streaming.api.windowing.triggers.Trigger;
import org.apache.flink.streaming.api.windowing.triggers.TriggerResult;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.api.windowing.windows.Window;

import java.text.SimpleDateFormat;

@PublicEvolving
public class BufferTrigger<W extends Window> extends Trigger<Object, W> {
    private static final long serialVersionUID = 1L;

    private final long maxCount;

    private final long interval;

    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");

    /** When merging we take the lowest of all fire timestamps as the new fire timestamp. */
    private final ReducingStateDescriptor<Long> countStateDesc =
            new ReducingStateDescriptor<>("count-fire-time", new BufferTrigger.Sum(), LongSerializer.INSTANCE);

    private final ReducingStateDescriptor<Long> timeoutStateDesc =
            new ReducingStateDescriptor<>("timeout-fire-time", new BufferTrigger.Min(), LongSerializer.INSTANCE);

    private BufferTrigger(long maxCount, long interval) {
        this.maxCount = maxCount;
        this.interval = interval;
    }

    @Override
    public TriggerResult onElement(Object element, long timestamp, W window, TriggerContext ctx) throws Exception {
        ReducingState<Long> fireCount = ctx.getPartitionedState(countStateDesc);
        ReducingState<Long> fireTimestamp = ctx.getPartitionedState(timeoutStateDesc);

        fireCount.add(1L);
        System.out.println("count: " + fireCount.get());
        //timestamp = ctx.getCurrentProcessingTime();

        if (fireCount.get() >= maxCount) {
            System.out.println("countTrigger triggered count: " + fireCount.get());
            fireCount.clear();
            if (fireTimestamp.get() != window.maxTimestamp()) {
                System.out.println("delete trigger: [" + fireTimestamp.get() + "|" + sdf.format(fireTimestamp.get()) + "]");
                ctx.deleteProcessingTimeTimer(fireTimestamp.get());
            }
            fireTimestamp.clear();
            return TriggerResult.FIRE;
        }

        timestamp = ctx.getCurrentProcessingTime();
        if (fireTimestamp.get() == null) {
            long start = timestamp - (timestamp % interval);
            long nextFireTimestamp = start + interval;
            ctx.registerProcessingTimeTimer(nextFireTimestamp);
            fireTimestamp.add(nextFireTimestamp);
        }

        return TriggerResult.CONTINUE;

//        if (fireTimestamp.get() == null) {
//            long start = timestamp - (timestamp % interval);
//            long nextFireTimestamp = start + interval;
//
//            ctx.registerProcessingTimeTimer(nextFireTimestamp);
//
//            fireTimestamp.add(nextFireTimestamp);
//            return TriggerResult.CONTINUE;
//        }
//        return TriggerResult.CONTINUE;
    }

    @Override
    public TriggerResult onEventTime(long time, W window, TriggerContext ctx) throws Exception {
        ReducingState<Long> fireCount = ctx.getPartitionedState(countStateDesc);
        if (time >= window.maxTimestamp() && fireCount.get() != null && fireCount.get() > 0L) {
            System.out.println(String.format("window.maxTimestamp: [%s|%s]", window.maxTimestamp(), sdf.format(window.maxTimestamp())));
            System.out.println(String.format("time: [%s|%s]", time, sdf.format(time)));
            System.out.println("triggered by eventTime");
            return TriggerResult.FIRE_AND_PURGE;
        }
        return TriggerResult.CONTINUE;
    }

    @Override
    public TriggerResult onProcessingTime(long time, W window, TriggerContext ctx) throws Exception {
//        ReducingState<Long> fireTimestamp = ctx.getPartitionedState(stateDesc);
//
//        if (fireTimestamp.get().equals(time)) {
//            fireTimestamp.clear();
//            fireTimestamp.add(time + interval);
//            ctx.registerProcessingTimeTimer(time + interval);
//            return TriggerResult.FIRE_AND_PURGE;
//        }
//        return TriggerResult.CONTINUE;
        ReducingState<Long> fireCount = ctx.getPartitionedState(countStateDesc);
        ReducingState<Long> fireTimestamp = ctx.getPartitionedState(timeoutStateDesc);
        if (time == window.maxTimestamp()) {
            System.out.println("window close: [" + time + "|" + sdf.format(time) + "]");
            fireCount.clear();
            ctx.deleteProcessingTimeTimer(fireTimestamp.get());
            fireTimestamp.clear();
            return TriggerResult.FIRE_AND_PURGE;
        } else if (fireTimestamp.get() != null && fireTimestamp.get().equals(time)) {
            System.out.println("timeTrigger triggered: [" + time + "|" + sdf.format(time) + "]");
            fireCount.clear();
            fireTimestamp.clear();
            return TriggerResult.FIRE;
        }

        return TriggerResult.CONTINUE;
    }

    @Override
    public void clear(W window, TriggerContext ctx) throws Exception {
        ReducingState<Long> fireTimestamp = ctx.getPartitionedState(timeoutStateDesc);
        long timestamp = fireTimestamp.get();
        ctx.deleteProcessingTimeTimer(timestamp);
        fireTimestamp.clear();
    }

    @Override
    public boolean canMerge() {
        return true;
    }

    @Override
    public void onMerge(W window,
                        OnMergeContext ctx) {
        ctx.mergePartitionedState(countStateDesc);
        ctx.mergePartitionedState(timeoutStateDesc);
    }

    @VisibleForTesting
    public long getInterval() {
        return interval;
    }

    @Override
    public String toString() {
        return "ContinuousProcessingTimeTrigger(" + interval + ")";
    }

    /**
     * Creates a trigger that continuously fires based on the given interval.
     *
     * @param interval The time interval at which to fire.
     * @param <W> The type of {@link Window Windows} on which this trigger can operate.
     */
    public static <W extends Window> BufferTrigger<W> of(long maxCount, Time interval) {
        return new BufferTrigger<>(maxCount, interval.toMilliseconds());
    }

    private static class Sum implements ReduceFunction<Long> {
        private static final long serialVersionUID = 1L;


        @Override
        public Long reduce(Long value1, Long value2) throws Exception {
            return value1 + value2;
        }
    }

    private static class Min implements ReduceFunction<Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long reduce(Long value1, Long value2) throws Exception {
            return Math.min(value1, value2);
        }
    }
}
