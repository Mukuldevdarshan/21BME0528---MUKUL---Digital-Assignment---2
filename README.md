import java.util.Random;

public class QLearning {
    private static final double ALPHA = 0.1; // Learning rate
    private static final double GAMMA = 0.99; // Discount factor
    private static final double EPSILON = 0.1; // Exploration rate
    private double[][] qTable;
    private Random random;

    public QLearning(int numStates, int numActions) {
        qTable = new double[numStates][numActions];
        random = new Random();
    }

    public int chooseAction(int state) {
        return random.nextDouble() < EPSILON ?
                random.nextInt(qTable[state].length) :
                maxIndex(qTable[state]);
    }

    public void updateQValue(int state, int action, double reward, int nextState) {
        qTable[state][action] += ALPHA * (reward + GAMMA * maxQValue(nextState) - qTable[state][action]);
    }

    private double maxQValue(int state) {
        return max(qTable[state]);
    }

    private int maxIndex(double[] array) {
        int index = 0;
        for (int i = 1; i < array.length; i++) if (array[i] > array[index]) index = i;
        return index;
    }

    private double max(double[] array) {
        double max = array[0];
        for (double value : array) if (value > max) max = value;
        return max;
    }

    public static void main(String[] args) {
        int numStates = 16, numActions = 4, numEpisodes = 1000, numTestEpisodes = 100;
        QLearning qLearning = new QLearning(numStates, numActions);

        for (int episode = 0; episode < numEpisodes; episode++) trainEpisode(qLearning, numStates);

        int successfulEpisodes = 0;
        for (int episode = 0; episode < numTestEpisodes; episode++)
            if (testEpisode(qLearning, numStates)) successfulEpisodes++;

        System.out.println("Success rate over " + numTestEpisodes + " test episodes: " +
                ((double) successfulEpisodes / numTestEpisodes) * 100 + "%");
    }

    private static void trainEpisode(QLearning qLearning, int numStates) {
        int state = 0;
        while (state != numStates - 1) {
            int action = qLearning.chooseAction(state);
            double reward = simulateEnvironment(state, action);
            int nextState = getNextState(state, action);
            qLearning.updateQValue(state, action, reward, nextState);
            state = nextState;
        }
    }

    private static boolean testEpisode(QLearning qLearning, int numStates) {
        int state = 0;
        while (state != numStates - 1) state = getNextState(state, qLearning.chooseAction(state));
        return state == numStates - 1;
    }

    private static double simulateEnvironment(int state, int action) {
        return (state == 14 && action == 3) ? 1 : 0;
    }

    private static int getNextState(int state, int action) {
        int[][] transitions = {{0, 1, 4, 0}, {1, 2, 5, 0}, {2, 3, 6, 1}, {3, 3, 7, 2}, {4, 0, 8, 5}, {5, 1, 9, 6},
                {6, 2, 10, 7}, {7, 3, 11, 3}, {8, 4, 12, 9}, {9, 5, 13, 10}, {10, 6, 14, 11}, {11, 7, 15, 7},
                {12, 8, 12, 13}, {13, 9, 13, 14}, {14, 10, 14, 15}, {15, 15, 15, 15}};
        return transitions[state][action];
    }
}
