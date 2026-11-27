import unittest

import pandas as pd

from hermae.splits import assign_splits, group_events


class GroupedSplitTest(unittest.TestCase):
    def test_prior_field_and_transitive_links_stay_in_training(self):
        events = pd.DataFrame(
            [
                ("overlap", 0.01, 1, 0.01, 1),
                ("shared-source", 40, 1, 40, 1),
                ("repeated-host", 80, 1, 40.0001, 1),
                ("same-region", 81, 1, 81, 1),
                ("other-a", 170, 1, 170, 1),
                ("other-b", 250, 1, 250, 1),
            ],
            columns=["event_id", "ra", "dec", "host_ra", "host_dec"],
        )
        development = pd.DataFrame([("prior", 359.99, 1, 359.99, 1)], columns=events.columns)
        candidates = pd.DataFrame([("overlap", "shared")], columns=["event_id", "source_id"])
        evaluation = pd.DataFrame([("shared-source", "shared")],
                                  columns=["event_id", "reference_source_id"])

        groups = group_events(events, evaluation, candidates, 360, 1.5, development)
        assignments = pd.DataFrame(assign_splits(groups, events.event_id, development.event_id, 42))
        anchored = assignments[assignments.event_id.isin(events.event_id.iloc[:4])]

        self.assertEqual(anchored.group_id.nunique(), 1)
        self.assertTrue(anchored.split.eq("train").all())
        self.assertEqual(set(assignments.split), {"train", "validation", "test"})
        shuffled = assign_splits(groups.sample(frac=1, random_state=3), events.event_id,
                                 development.event_id, 42)
        self.assertEqual(assignments.to_dict("records"), shuffled)


if __name__ == "__main__":
    unittest.main()
