clear


% define the start and end dates
start_date = datetime( 2005,01,03,00,30,00 );
end_date = datetime(2022,07,15,00,00,00 );

% create a vector of all minute timestamps for the specified days
timestamps = start_date:minutes(1):end_date+days(1)-minutes(1);

% create a timetable with the timestamps as row times
timetable_data = table('Size',[numel(timestamps),1],'VariableTypes',{'double'});
timetable_data.Properties.VariableNames = {'MyData'};
timetable_data.MyData = randn(numel(timestamps),1);
my_timetable = timetable(timetable_data,'RowTimes',timestamps);


% define the start and end times for the filtering period
start_time = timeofday(datetime('09:30:00'));
end_time = timeofday(datetime('16:00:00'));

% create a logical index for the rows that fall between the start and end times
time_filter = (timeofday(my_timetable.Properties.RowTimes) >= start_time) & ...
              (timeofday(my_timetable.Properties.RowTimes) <= end_time);

% filter the rows of the timetable using the logical index
filtered_timetable = my_timetable(time_filter,:);

% create a logical index for the business days
business_days = isbusday(filtered_timetable.Properties.RowTimes);

% remove the rows corresponding to business days from the timetable
filtered_timetable = filtered_timetable(business_days, :);



