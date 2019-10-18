require "csv"

class Array
  def r(y)
    raise "Argument is not a Array class!"  unless y.class == Array
    raise "Self array is nil!"              if self.size == 0
    raise "Argument array size is invalid!" unless self.size == y.size

    mean_x = self.inject(0) { |s, a| s += a } / self.size.to_f
    mean_y = y.inject(0) { |s, a| s += a } / y.size.to_f

    cov = self.zip(y).inject(0) { |s, a| s += (a[0] - mean_x) * (a[1] - mean_y) }

    var_x = self.inject(0) { |s, a| s += (a - mean_x) ** 2 }
    var_y = y.inject(0) { |s, a| s += (a - mean_y) ** 2 }

    r = cov / Math.sqrt(var_x)
    r /= Math.sqrt(var_y)
  end
end

class Parser
  def initialize(filename)
    @parsed_file = CSV.read(filename, { col_sep: ' ' })
    @result = []
    n_cols = @parsed_file[0].size
    # 378 проходов - для 28 столбцов (на моей машине отрабатывает за 0.2с)
    (0 ... n_cols).to_a.combination(2).to_a.each do |(n_col1, n_col2)|
      col1, col2 = [], []
      @parsed_file.each do |row|
        val1, val2 = row[n_col1], row[n_col2]
        next if val1 == '?' || val2 == '?'
        col1 << val1.to_f
        col2 << val2.to_f
      end
      r = col1.r(col2)
      @result << [r, n_col1 + 1, n_col2 + 1] if r && !r.nan?
    end
    @result.sort{ |x, y| x[0].abs <=> y[0].abs }.each { |x| p x }
  end

  def get_column(n)
    col = []
    @parsed_file.each { |row| col << row[n - 1] }
    col
  end
end

parser = Parser.new('horse-colic.data')
