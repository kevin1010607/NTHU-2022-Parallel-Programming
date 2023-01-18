/***************************************************************
*SortComparator: SecondarySortBasicCompKeySortComparator
*****************************************************************/

import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.WritableComparable;

public class SecondarySortBasicCompKeySortComparator extends WritableComparator {

  protected SecondarySortBasicCompKeySortComparator() {
		super(CompositeKeyWritable.class, true);
	}

	@Override
	public int compare(WritableComparable w1, WritableComparable w2) {
		CompositeKeyWritable key1 = (CompositeKeyWritable) w1;
		CompositeKeyWritable key2 = (CompositeKeyWritable) w2;

		int cmpResult = key1.getDeptNo().compareTo(key2.getDeptNo());
		if (cmpResult == 0)// same deptNo
		{
			return -key1.getLNameEmpIDPair()
					.compareTo(key2.getLNameEmpIDPair());
			//If the minus is taken out, the values will be in
			//ascending order
		}
		return cmpResult;
	}
}
